"""
04_flight_error_analysis.py
Hangi uçuş koşulları GRU hatasını artırıyor?
Her test uçuşu için: hız, manevra şiddeti, uçuş süresi vs HPE korelasyonu.
"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE    = Path(r"c:\Users\Lenovo\Desktop\projeler egit\SSBHava Araçlarında GPS Kullanmadan Konum Tahmini")
PH1     = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2     = BASE / "Phase2_Model" / "outputs"
SCRIPTS = BASE / "Phase2_Model" / "scripts"
OUT_DIR = BASE / "Phase4_Rapor" / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS))
from models import GRUModel, detrend_windows

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru = GRUModel().to(device)
gru.load_state_dict(torch.load(PH2 / "best_gru.pt", weights_only=True, map_location=device))
gru.eval()

SEQ_LEN = 40
STEP    = 4
FS      = 2   # Hz

def make_windows(X_raw):
    windows = []
    for i in range(0, len(X_raw) - SEQ_LEN + 1, STEP):
        windows.append(X_raw[i:i+SEQ_LEN])
    return np.stack(windows).astype(np.float32)

# ── Her test uçuşu için özellik hesapla ─────────────────────────────────────
test_npzs = sorted(PH1.glob("flight_*_test.npz"))
print(f"Test uçuşu sayısı: {len(test_npzs)}")

records = []
for npz_path in test_npzs:
    d = np.load(npz_path)
    X_raw   = d["X_scaled"]    # (N, 12)
    y_delta = d["y_delta"]     # (N, 3)
    time_s  = d["time_s"]      # (N,)

    # Uçuş özellikleri
    n_steps    = len(X_raw)
    dur_s      = float(time_s[-1] - time_s[0])
    mean_speed = float(np.mean(np.abs(X_raw[:, 9])))          # airspeed
    gyro_std   = float(np.mean(np.std(X_raw[:, :3], axis=0))) # gyro değişkenliği = manevra şiddeti
    accel_std  = float(np.mean(np.std(X_raw[:, 3:6], axis=0)))# ivme değişkenliği
    mag_std    = float(np.mean(np.std(X_raw[:, 6:9], axis=0)))# mag değişkenliği

    # GRU inference — windows
    windows = make_windows(X_raw)
    if len(windows) == 0:
        continue
    wins_dt = detrend_windows(windows)
    t = torch.from_numpy(wins_dt).to(device)
    with torch.no_grad():
        preds = gru(t).cpu().numpy()   # (W, 3)

    # Pencere adımlarına karşılık gelen gerçek değerleri al
    step_idxs = [min(i * STEP + SEQ_LEN - 1, n_steps - 1) for i in range(len(preds))]
    y_true_w  = y_delta[step_idxs]      # (W, 3)

    err = preds - y_true_w
    hpe = np.sqrt(err[:, 0]**2 + err[:, 1]**2)
    mean_hpe  = float(np.mean(hpe))
    median_hpe= float(np.median(hpe))
    p95_hpe   = float(np.percentile(hpe, 95))

    records.append({
        "flight":      npz_path.stem.replace("flight_", "").replace("_test", ""),
        "dur_s":       dur_s,
        "n_steps":     n_steps,
        "n_windows":   len(preds),
        "mean_speed":  mean_speed,
        "gyro_std":    gyro_std,
        "accel_std":   accel_std,
        "mag_std":     mag_std,
        "HPE_mean":    mean_hpe,
        "HPE_median":  median_hpe,
        "HPE_p95":     p95_hpe,
    })

# JSON kaydet
with open(BASE / "Phase4_Rapor" / "outputs" / "flight_error_analysis.json", "w") as f:
    json.dump(records, f, indent=2)

# ── Korelasyon grafikleri ────────────────────────────────────────────────────
features = [
    ("dur_s",      "Uçuş Süresi (s)",         "b"),
    ("mean_speed", "Ortalama Hız (scaled)",    "g"),
    ("gyro_std",   "Gyro Std (manevra şiddeti)","r"),
    ("accel_std",  "İvme Std",                 "purple"),
    ("mag_std",    "Mag Std",                  "orange"),
    ("n_windows",  "Pencere Sayısı",           "brown"),
]

hpe_means = np.array([r["HPE_mean"] for r in records])

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Uçuş Koşulları vs GRU Ortalama HPE\n"
             "Hangi koşullar navigasyon hatasını artırıyor?",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

corr_results = []
for ax, (feat, label, color) in zip(axes, features):
    x_vals = np.array([r[feat] for r in records])
    corr   = float(np.corrcoef(x_vals, hpe_means)[0, 1])
    corr_results.append((feat, label, corr))

    ax.scatter(x_vals, hpe_means, color=color, s=60, alpha=0.8, edgecolors="black", linewidths=0.5)
    # Trend çizgisi
    z = np.polyfit(x_vals, hpe_means, 1)
    p = np.poly1d(z)
    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xs, p(xs), "--", color=color, alpha=0.6, linewidth=1.5)

    # Etiketler
    for r in records:
        ax.annotate(r["flight"].replace("vuelo_", ""),
                    (r[feat], r["HPE_mean"]),
                    textcoords="offset points", xytext=(3, 3), fontsize=6, alpha=0.7)

    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("HPE Ortalama (m)", fontsize=9)
    corr_color = "red" if abs(corr) > 0.5 else ("darkorange" if abs(corr) > 0.3 else "black")
    ax.set_title(f"Pearson r = {corr:.3f}", fontsize=10, fontweight="bold", color=corr_color)
    ax.grid(alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "flight_error_correlation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Grafik: {out}")

# ── HPE sıralaması ───────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 5))
sorted_r = sorted(records, key=lambda r: r["HPE_mean"])
names    = [r["flight"].replace("vuelo_","v") for r in sorted_r]
means    = [r["HPE_mean"] for r in sorted_r]
p95s     = [r["HPE_p95"] for r in sorted_r]
colors_bar = ["#F44336" if m > 6 else ("#FF9800" if m > 4 else "#4CAF50") for m in means]

bars = ax.bar(range(len(names)), means, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
ax.plot(range(len(names)), p95s, "D--", color="navy", markersize=5, linewidth=1.2, label="HPE P95")
ax.axhline(np.mean(means), color="red", linestyle=":", linewidth=2,
           label=f"Ortalama={np.mean(means):.2f}m")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("HPE (m)", fontsize=10)
ax.set_title("Test Uçuşları — GRU Per-Adım HPE Sıralaması\n(Yeşil<4m, Turuncu 4-6m, Kırmızı>6m)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out2 = OUT_DIR / "per_flight_hpe_ranking.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Grafik: {out2}")

# ── Özet ────────────────────────────────────────────────────────────────────
print()
print("=== Korelasyon Özeti ===")
print(f"{'Özellik':<35} {'Pearson r':>10} {'Yorum'}")
print("-"*65)
for feat, label, corr in sorted(corr_results, key=lambda x: abs(x[2]), reverse=True):
    strength = "GÜÇLÜ" if abs(corr) > 0.5 else ("ORTA" if abs(corr) > 0.3 else "zayıf")
    direction = "pozitif" if corr > 0 else "negatif"
    print(f"{label:<35} {corr:>+10.3f}  {strength} {direction}")

print()
print(f"En kolay uçuş : {sorted_r[0]['flight']}  HPE={sorted_r[0]['HPE_mean']:.3f}m")
print(f"En zor uçuş   : {sorted_r[-1]['flight']} HPE={sorted_r[-1]['HPE_mean']:.3f}m")
