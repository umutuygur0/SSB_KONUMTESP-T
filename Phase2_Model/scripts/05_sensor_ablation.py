"""
05_sensor_ablation.py
Sensör ablasyon analizi: belirli sensör grupları sıfırlandığında GRU HPE'ye etkisi.
Hangi sensör grubu olmadan model ne kadar kötüleşiyor?

Özellik dizini (12 feature):
  0-2  : delta_angle[0-2]   → IMU açısal hız (gyro)
  3-5  : delta_velocity[0-2] → IMU hız (ivmeölçer)
  6-8  : mag_field[0-2]      → Manyetometre
  9    : indicated_airspeed   → Hava hızı
  10   : baro_alt_meter       → Barometrik irtifa
  11   : differential_pressure→ Diferansiyel basınç
"""
import importlib.util, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE    = Path(r"c:\Users\Lenovo\Desktop\projeler egit\SSBHava Araçlarında GPS Kullanmadan Konum Tahmini")
PH1     = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2     = BASE / "Phase2_Model" / "outputs"
SCRIPTS = BASE / "Phase2_Model" / "scripts"

_spec = importlib.util.spec_from_file_location("models", SCRIPTS / "models.py")
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
detrend_windows = _mod.detrend_windows
GRUModel        = _mod.GRUModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru = GRUModel().to(device)
gru.load_state_dict(torch.load(PH2 / "best_gru.pt", weights_only=True, map_location=device))
gru.eval()

# Test seti
X_te = np.load(PH1 / "X_test.npy")   # (2470, 40, 12) — zaten pencerelenmiş
y_te = np.load(PH1 / "y_test.npy")   # (2470, 3)

def compute_hpe(X_windows, y_true, zero_cols=None):
    """Belirli sütunlar sıfırlanmış halde HPE hesapla."""
    X_copy = X_windows.copy()
    if zero_cols:
        X_copy[:, :, zero_cols] = 0.0

    X_dt = detrend_windows(X_copy)
    t    = torch.from_numpy(X_dt).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(t), 512):
            preds.append(gru(t[i:i+512]).cpu().numpy())
    yp  = np.vstack(preds)
    err = yp - y_true
    hpe = np.sqrt(err[:, 0]**2 + err[:, 1]**2)
    return float(np.mean(hpe)), float(np.median(hpe)), float(np.percentile(hpe, 95))

# ── Ablasyon senaryoları ─────────────────────────────────────────────────────
scenarios = [
    ("Tam Model\n(Baseline)",   None,          "#2196F3"),
    ("Mag. yok\n(cols 6-8)",   [6, 7, 8],      "#FF9800"),
    ("Baro yok\n(col 10)",     [10],            "#4CAF50"),
    ("Airspeed yok\n(cols 9,11)",[9, 11],       "#9C27B0"),
    ("Sadece IMU\n(cols 0-5)", [6,7,8,9,10,11], "#F44336"),
]

print("Sensör Ablasyon Analizi çalışıyor...")
results = []
for label, zero_cols, color in scenarios:
    mean_hpe, med_hpe, p95_hpe = compute_hpe(X_te, y_te, zero_cols)
    results.append({
        "scenario":  label.replace("\n", " "),
        "zero_cols": zero_cols,
        "HPE_mean":  round(mean_hpe, 3),
        "HPE_median": round(med_hpe, 3),
        "HPE_p95":   round(p95_hpe, 3),
    })
    print(f"  {label.replace(chr(10),' '):<30}  HPE_mean={mean_hpe:.3f}m  med={med_hpe:.3f}m  p95={p95_hpe:.3f}m")

baseline_mean = results[0]["HPE_mean"]
for r in results:
    r["delta_vs_baseline"] = round(r["HPE_mean"] - baseline_mean, 3)
    r["pct_increase"]      = round((r["HPE_mean"] - baseline_mean) / baseline_mean * 100, 1)

# ── Grafik ──────────────────────────────────────────────────────────────────
labels   = [r["scenario"].replace(" ","\n") if "\n" not in r["scenario"] else r["scenario"] for r in results]
# Preserve original labels with newlines
labels   = [s[0] for s in scenarios]
means    = [r["HPE_mean"]   for r in results]
p95s     = [r["HPE_p95"]    for r in results]
colors   = [s[2]            for s in scenarios]
deltas   = [r["delta_vs_baseline"] for r in results]

x = np.arange(len(scenarios))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Sensör Ablasyon Analizi — GRU Modeli\nHangi sensör grubu kaldırıldığında HPE nasıl değişir?",
             fontsize=13, fontweight="bold")

# Bar 1: HPE Ortalama
ax = axes[0]
bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
ax.axhline(baseline_mean, color="navy", linestyle="--", linewidth=1.5, label=f"Baseline: {baseline_mean:.3f}m")
for i, (b, m, d) in enumerate(zip(bars, means, deltas)):
    sign = "+" if d >= 0 else ""
    ax.text(b.get_x() + b.get_width()/2, m + 0.05, f"{m:.2f}m\n({sign}{d:.2f})",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("HPE Ortalama (m)", fontsize=10)
ax.set_title("Per-adım HPE Ortalama", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, max(means) * 1.30)
ax.grid(axis="y", alpha=0.3)

# Bar 2: HPE P95
ax2 = axes[1]
bars2 = ax2.bar(x, p95s, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
baseline_p95 = results[0]["HPE_p95"]
ax2.axhline(baseline_p95, color="navy", linestyle="--", linewidth=1.5, label=f"Baseline p95: {baseline_p95:.2f}m")
for b, p in zip(bars2, p95s):
    ax2.text(b.get_x() + b.get_width()/2, p + 0.1, f"{p:.1f}m",
             ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("HPE P95 (m)", fontsize=10)
ax2.set_title("Per-adım HPE P95 (95. yüzdelik)", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(p95s) * 1.20)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_png = PH2 / "plots" / "sensor_ablation_hpe.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nGrafik: {out_png}")

# JSON
out_json = PH2 / "sensor_ablation.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"JSON : {out_json}")

print("\n=== Özet ===")
print(f"{'Senaryo':<35} {'HPE_mean':>10} {'Δ':>8} {'%Artış':>8}")
print("-"*65)
for r in results:
    sign = "+" if r['delta_vs_baseline'] >= 0 else ""
    print(f"{r['scenario']:<35} {r['HPE_mean']:>10.3f}m {sign+str(r['delta_vs_baseline']):>8} {r['pct_increase']:>7.1f}%")
