"""
Phase 2 — Adım 1: Dead Reckoning Baseline
delta_velocity'yi DT ile çarparak basit NED tahmini yapar.
Gerçek modeller için karşılaştırma referansı.
"""

import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]
PHASE1_OUT = BASE_DIR / "Phase1_Veri_Analizi" / "outputs"
PHASE2_OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS_DIR  = PHASE2_OUT / "plots"
PHASE2_OUT.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Feature sırası (Phase1 preprocessing'deki INPUT_COLS ile aynı)
# [da0, da1, da2, dv0, dv1, dv2, mag0, mag1, mag2, airspeed, baro_alt, diff_press]
DV_IDX = slice(3, 6)   # delta_velocity kolonları
DT = 0.5               # 2 Hz → 0.5 s/adım


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """ΔNorth, ΔEast, ΔUp için tüm metrikleri hesapla."""
    err  = y_true - y_pred
    dn, de, du = err[:, 0], err[:, 1], err[:, 2]
    hpe  = np.sqrt(dn**2 + de**2)
    e3d  = np.sqrt(dn**2 + de**2 + du**2)
    rmse = np.sqrt((err**2).mean(axis=0))
    mae  = np.abs(err).mean(axis=0)
    return {
        "HPE_mean":    float(hpe.mean()),
        "HPE_median":  float(np.median(hpe)),
        "HPE_p95":     float(np.percentile(hpe, 95)),
        "3DE_mean":    float(e3d.mean()),
        "3DE_median":  float(np.median(e3d)),
        "RMSE_north":  float(rmse[0]),
        "RMSE_east":   float(rmse[1]),
        "RMSE_up":     float(rmse[2]),
        "RMSE_3D":     float(np.sqrt((err**2).mean())),
        "MAE_north":   float(mae[0]),
        "MAE_east":    float(mae[1]),
        "MAE_up":      float(mae[2]),
        "n_samples":   int(len(y_true)),
    }


def dead_reckoning_predict(X_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    X_scaled: (N, 40, 12)  — pencereli, normalize edilmiş veri
    Son adımın delta_velocity'sini ters-normalize edip DT ile çarpar.
    Çıkış: (N, 3)  [ΔNorth, ΔEast, ΔUp]
    Varsayım: body_x ≈ North, body_y ≈ East, body_z ≈ Down (seviyeli uçuş)
    """
    X_last_sc  = X_scaled[:, -1, :]              # (N, 12)
    X_last_raw = scaler.inverse_transform(X_last_sc)  # (N, 12)
    dv = X_last_raw[:, DV_IDX]                   # (N, 3) — m/s
    return np.column_stack([
        dv[:, 0] * DT,    # ΔNorth ≈ dv_x * dt
        dv[:, 1] * DT,    # ΔEast  ≈ dv_y * dt
       -dv[:, 2] * DT,    # ΔUp    ≈ -dv_z * dt  (z_body = Down)
    ])


def plot_error_distributions(y_true, y_pred, title_prefix="Dead Reckoning"):
    err = y_true - y_pred
    dn, de, du = err[:, 0], err[:, 1], err[:, 2]
    hpe = np.sqrt(dn**2 + de**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    kw = dict(bins=60, edgecolor='black', linewidth=0.4)

    axes[0, 0].hist(dn, alpha=0.75, color='steelblue', **kw)
    axes[0, 0].axvline(0, color='r', ls='--', alpha=0.8)
    axes[0, 0].set_title("ΔNorth Hatası (m)")
    axes[0, 0].set_xlabel("Hata (m)")

    axes[0, 1].hist(de, alpha=0.75, color='darkorange', **kw)
    axes[0, 1].axvline(0, color='r', ls='--', alpha=0.8)
    axes[0, 1].set_title("ΔEast Hatası (m)")
    axes[0, 1].set_xlabel("Hata (m)")

    axes[1, 0].hist(du, alpha=0.75, color='seagreen', **kw)
    axes[1, 0].axvline(0, color='r', ls='--', alpha=0.8)
    axes[1, 0].set_title("ΔUp Hatası (m)")
    axes[1, 0].set_xlabel("Hata (m)")

    med = np.median(hpe)
    p95 = np.percentile(hpe, 95)
    axes[1, 1].hist(hpe, alpha=0.75, color='mediumpurple', **kw)
    axes[1, 1].axvline(med, color='r', ls='--', alpha=0.9, label=f"Medyan={med:.2f}m")
    axes[1, 1].axvline(p95, color='orange', ls='-.', alpha=0.9, label=f"P95={p95:.2f}m")
    axes[1, 1].set_title("Yatay Konum Hatası — HPE (m)")
    axes[1, 1].set_xlabel("HPE (m)")
    axes[1, 1].legend(fontsize=9)

    for ax in axes.flat:
        ax.set_ylabel("Frekans")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{title_prefix} — Hata Dağılımları", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = PLOTS_DIR / "baseline_error_dist.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out}")


def main():
    print("=" * 60)
    print("ADIM 1: Dead Reckoning Baseline")
    print("=" * 60)

    X_test = np.load(PHASE1_OUT / "X_test.npy")
    y_test = np.load(PHASE1_OUT / "y_test.npy")
    with open(PHASE1_OUT / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"X_test: {X_test.shape},  y_test: {y_test.shape}")

    y_pred = dead_reckoning_predict(X_test, scaler)
    metrics = compute_metrics(y_test, y_pred)

    print("\nDead Reckoning Metrikleri (test seti):")
    print(f"  HPE  mean / median / P95 : {metrics['HPE_mean']:.3f} / {metrics['HPE_median']:.3f} / {metrics['HPE_p95']:.3f} m")
    print(f"  3DE  mean / median       : {metrics['3DE_mean']:.3f} / {metrics['3DE_median']:.3f} m")
    print(f"  RMSE (N, E, Up, 3D)      : {metrics['RMSE_north']:.3f} / {metrics['RMSE_east']:.3f} / {metrics['RMSE_up']:.3f} / {metrics['RMSE_3D']:.3f} m")
    print(f"  MAE  (N, E, Up)          : {metrics['MAE_north']:.3f} / {metrics['MAE_east']:.3f} / {metrics['MAE_up']:.3f} m")

    out_json = PHASE2_OUT / "metrics_baseline.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrikler: {out_json}")

    plot_error_distributions(y_test, y_pred)

    print("\nAdım 1 tamamlandı.\n")
    return metrics


if __name__ == "__main__":
    main()
