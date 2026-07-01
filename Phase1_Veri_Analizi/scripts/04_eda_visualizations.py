"""
Phase 1 — EDA Görselleri
1. Feature dağılımları (train/val/test karşılaştırması)
2. Feature–Target korelasyon ısı haritası
3. Veri kalitesi: uçuş başına örnek sayısı ve süre dağılımı
4. Mag/baro dağılım anomalisi (neden detrend gerekli?)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE    = Path(r"c:\Users\Lenovo\Desktop\projeler egit\SSBHava Araçlarında GPS Kullanmadan Konum Tahmini")
PH1_OUT = BASE / "Phase1_Veri_Analizi" / "outputs"
OUT_DIR = BASE / "Phase1_Veri_Analizi" / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "ΔAngle X (gyro)", "ΔAngle Y (gyro)", "ΔAngle Z (gyro)",
    "ΔVel X (accel)",  "ΔVel Y (accel)",  "ΔVel Z (accel)",
    "Mag X",           "Mag Y",            "Mag Z",
    "Airspeed",        "Baro Alt",         "Diff Pressure",
]
TARGET_NAMES = ["ΔNorth", "ΔEast", "ΔUp"]

# ── Veri yükle ───────────────────────────────────────────────────────────────
X_tr = np.load(PH1_OUT / "X_train.npy")  # (9677, 40, 12)
X_vl = np.load(PH1_OUT / "X_val.npy")
X_te = np.load(PH1_OUT / "X_test.npy")
y_tr = np.load(PH1_OUT / "y_train.npy")  # (9677, 3)

# Son timestep (pencere sonu) değerlerini al — dağılım analizi için
Xtr_last = X_tr[:, -1, :]  # (N, 12)
Xvl_last = X_vl[:, -1, :]
Xte_last = X_te[:, -1, :]

print(f"Train: {X_tr.shape}  Val: {X_vl.shape}  Test: {X_te.shape}")

# ════════════════════════════════════════════════════════════════════════
# GRAFİK 1: Feature Dağılımları — Train vs Val vs Test
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Girdi Özelliği Dağılımları — Train / Val / Test Seti Karşılaştırması",
             fontsize=13, fontweight="bold", y=1.01)
axes = axes.flatten()

for i, (ax, name) in enumerate(zip(axes, FEATURE_NAMES)):
    data_tr = Xtr_last[:, i]
    data_vl = Xvl_last[:, i]
    data_te = Xte_last[:, i]

    # Box plot — medyan, IQR, aykırı değerler
    bp = ax.boxplot([data_tr, data_vl, data_te],
                    labels=["Train", "Val", "Test"],
                    patch_artist=True, notch=False, showfliers=False)
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    # Std değerleri metin olarak
    stds = [data_tr.std(), data_vl.std(), data_te.std()]
    ratio = stds[2] / (stds[0] + 1e-9)
    title_color = "red" if ratio > 5 else "black"
    ax.set_title(f"{name}\nStd: Tr={stds[0]:.3f} | Te={stds[2]:.3f} (×{ratio:.1f})",
                 fontsize=8, color=title_color)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "feature_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Kaydedildi: {out}")

# ════════════════════════════════════════════════════════════════════════
# GRAFİK 2: Feature–Target Korelasyon Isı Haritası
# ════════════════════════════════════════════════════════════════════════
# Son timestep features + targets
all_data = np.hstack([Xtr_last, y_tr])  # (N, 15)
all_names = FEATURE_NAMES + TARGET_NAMES
corr = np.corrcoef(all_data.T)          # (15, 15)

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(15)); ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(15)); ax.set_yticklabels(all_names, fontsize=8)

for i in range(15):
    for j in range(15):
        val = corr[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)

# Hedef bölgesini vurgula
rect = plt.Rectangle((11.5, -0.5), 3, 15, fill=False, edgecolor="gold", linewidth=2.5, linestyle="--")
ax.add_patch(rect)
rect2 = plt.Rectangle((-0.5, 11.5), 15, 3, fill=False, edgecolor="gold", linewidth=2.5, linestyle="--")
ax.add_patch(rect2)

plt.colorbar(im, ax=ax, fraction=0.03, label="Pearson r")
ax.set_title("Özellik–Hedef Korelasyon Matrisi (Train Seti)\n"
             "Altın çerçeve = hedef değişkenler (ΔNorth, ΔEast, ΔUp)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
out = OUT_DIR / "feature_correlation_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Kaydedildi: {out}")

# ════════════════════════════════════════════════════════════════════════
# GRAFİK 3: Veri Kalitesi — Uçuş bazında örneklem ve süre
# ════════════════════════════════════════════════════════════════════════
import json
npz_files = sorted(PH1_OUT.glob("flight_*.npz"))
flight_info = []
for f in npz_files:
    d = np.load(f)
    n_steps = len(d["X_raw"])
    dur_s   = float(d["time_s"][-1] - d["time_s"][0]) if "time_s" in d else n_steps * 0.5
    split   = "train" if "_train" in f.name else ("val" if "_val" in f.name else "test")
    flight_info.append({"name": f.stem, "n_steps": n_steps, "dur_s": dur_s, "split": split})

splits  = {"train": [], "val": [], "test": []}
for fi in flight_info:
    splits[fi["split"]].append(fi)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Uçuş Veri Kalitesi Analizi", fontsize=13, fontweight="bold")

colors_split = {"train": "#4CAF50", "val": "#2196F3", "test": "#FF9800"}

# Sol: Süre dağılımı
ax = axes[0]
for split, items in splits.items():
    durs = [fi["dur_s"] for fi in items]
    ax.hist(durs, bins=12, label=f"{split} (n={len(items)})", alpha=0.7, color=colors_split[split])
ax.set_xlabel("Uçuş Süresi (saniye)"); ax.set_ylabel("Uçuş Sayısı")
ax.set_title("Uçuş Süresi Dağılımı"); ax.legend(); ax.grid(alpha=0.3)

# Orta: Adım sayısı dağılımı
ax = axes[1]
for split, items in splits.items():
    steps = [fi["n_steps"] for fi in items]
    ax.hist(steps, bins=12, label=f"{split} (n={len(items)})", alpha=0.7, color=colors_split[split])
ax.set_xlabel("Adım Sayısı (satır)"); ax.set_ylabel("Uçuş Sayısı")
ax.set_title("Uçuş Başına Adım Sayısı"); ax.legend(); ax.grid(alpha=0.3)

# Sağ: Toplam veri özeti
ax = axes[2]
splits_data = {"Küme": ["Train", "Val", "Test"],
               "Uçuş": [len(splits["train"]), len(splits["val"]), len(splits["test"])],
               "Toplam Adım": [sum(fi["n_steps"] for fi in v) for v in splits.values()]}
step_vals = [s//1000 for s in splits_data["Toplam Adım"]]
bars = ax.bar(splits_data["Küme"], step_vals,
              color=[colors_split[k] for k in ["train","val","test"]], alpha=0.85, edgecolor="black")
for bar, n_f, n_s in zip(bars, splits_data["Uçuş"], splits_data["Toplam Adım"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{n_s:,} adım\n({n_f} uçuş)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Toplam Adım (×1000)")
ax.set_title("Küme Başına Toplam Veri")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "data_quality_overview.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Kaydedildi: {out}")

# ════════════════════════════════════════════════════════════════════════
# GRAFİK 4: Mag/Baro Anomalisi — neden detrend gerekli?
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Manyetometre & Barometre Dağılım Anomalisi\n"
             "Neden per-window z-score (detrend_windows) uygulanıyor?",
             fontsize=12, fontweight="bold")

mag_cols  = [6, 7, 8]
baro_col  = 10
col_info  = [(6, "Mag X"), (7, "Mag Y"), (8, "Mag Z"), (10, "Baro Alt")]

for idx, (col, name) in enumerate(col_info):
    ax = axes[idx // 2][idx % 2]
    tr_data = Xtr_last[:, col]
    te_data = Xte_last[:, col]
    ratio   = te_data.std() / (tr_data.std() + 1e-9)
    ax.hist(tr_data, bins=60, alpha=0.6, color="#4CAF50", label=f"Train (std={tr_data.std():.3f})",
            density=True)
    ax.hist(te_data, bins=60, alpha=0.6, color="#FF5722", label=f"Test  (std={te_data.std():.3f})",
            density=True)
    ax.set_title(f"{name} — Test/Train Std Oranı: ×{ratio:.1f}", fontweight="bold",
                 color="red" if ratio > 5 else "darkorange")
    ax.set_xlabel("Ölçeklenmiş Değer"); ax.set_ylabel("Yoğunluk")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "mag_baro_anomaly.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Kaydedildi: {out}")

print()
print("=== EDA Özet ===")
total_steps = sum(fi["n_steps"] for fi in flight_info)
print(f"Toplam uçuş    : {len(flight_info)}")
print(f"Toplam adım    : {total_steps:,}")
print(f"Train/Val/Test : {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} uçuş")
print(f"Mag X std oranı (test/train): {Xte_last[:,6].std()/Xtr_last[:,6].std():.1f}x")
print(f"Baro  std oranı (test/train): {Xte_last[:,10].std()/Xtr_last[:,10].std():.1f}x")
print("4 EDA grafiği oluşturuldu.")
