"""
Phase 4 – 01: Trajektori Grafikleri
  - Tam uçuş: model tahmini vs GPS ground truth (kümülatif)
  - GPS kesintisi: kesinti öncesi gerçek, sonrası model tahmini
  - 2D (North-East) + 3D (North-East-Alt) grafikleri
"""

import numpy as np
import torch
import torch.nn as nn
import importlib.util
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

BASE     = Path(__file__).resolve().parents[2]
PH1      = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2      = BASE / "Phase2_Model" / "outputs"
SCRIPTS2 = BASE / "Phase2_Model" / "scripts"
PH4_PLOT = Path(__file__).resolve().parents[1] / "outputs" / "plots"

_spec = importlib.util.spec_from_file_location("models", SCRIPTS2 / "models.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
GRUModel = _m.GRUModel

SEQ_LEN = 40
FS      = 2    # Hz


def load_gru(device):
    m = GRUModel().to(device)
    m.load_state_dict(torch.load(PH2 / "best_gru.pt", weights_only=True))
    m.eval()
    return m


def make_windows(X_scaled):
    N = len(X_scaled)
    windows = np.zeros((N, SEQ_LEN, 12), dtype=np.float32)
    for t in range(N):
        s = max(0, t - SEQ_LEN + 1)
        w = X_scaled[s:t+1, :]
        if len(w) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(w), 12), dtype=np.float32)
            w = np.concatenate([pad, w], axis=0)
        windows[t] = w
    return windows


def detrend_window(window_np):
    x = window_np.copy()
    for c in [6, 7, 8, 10]:
        col = x[:, c]
        x[:, c] = (col - col.mean()) / (col.std() + 1e-8)
    return x


@torch.no_grad()
def run_inference(model, X_scaled, device, batch=256):
    windows = make_windows(X_scaled)
    preds = []
    for i in range(0, len(windows), batch):
        batch_w = np.stack([detrend_window(w) for w in windows[i:i+batch]])
        t = torch.from_numpy(batch_w).to(device)
        preds.append(model(t).cpu().numpy())
    return np.vstack(preds)   # (N, 3) — ΔNorth, ΔEast, ΔUp


def reconstruct_trajectory(y_abs_0, y_delta_pred):
    """y_abs_0: başlangıç konumu (2 veya 3,), y_delta_pred: (N, 2 veya 3)"""
    pos = np.zeros((len(y_delta_pred) + 1, y_delta_pred.shape[1]))
    pos[0] = y_abs_0[:y_delta_pred.shape[1]]
    for t in range(len(y_delta_pred)):
        pos[t+1] = pos[t] + y_delta_pred[t]
    return pos   # (N+1, dim)


# ── 2D Rota Grafikleri ──────────────────────────────────────────────────
def plot_2d_trajectory(flights_data, device, model):
    """4 örnek uçuş: gerçek vs GRU kümülatif tahmin"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (npz_path, X_scaled, y_delta, y_abs, _) in zip(axes, flights_data):
        preds = run_inference(model, X_scaled, device)

        # True trajectory (GPS)
        true_north = y_abs[:, 0]
        true_east  = y_abs[:, 1]

        # Model cumulative reconstruction from y_abs[0]
        pos = reconstruct_trajectory(y_abs[0, :2], preds[:, :2])
        pred_north = pos[1:, 0]   # skip başlangıç (pred starts from step 1)
        pred_east  = pos[1:, 1]

        name = npz_path.stem.replace("flight_", "").replace("_test", "")
        ax.plot(true_east, true_north, "b-", lw=1.5, label="GPS (Gerçek)")
        ax.plot(pred_east, pred_north, "r--", lw=1.5, alpha=0.8, label="GRU Tahmin")
        ax.plot(true_east[0], true_north[0], "go", ms=8, label="Başlangıç")
        ax.plot(true_east[-1], true_north[-1], "rs", ms=8, label="Bitiş")
        ax.set(xlabel="East (m)", ylabel="North (m)", title=f"{name}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_aspect("equal")

        # HPE metrigi
        hpe = np.sqrt((pred_north - true_north)**2 + (pred_east - true_east)**2)
        ax.text(0.02, 0.98, f"HPE ort={np.mean(hpe):.1f}m\nHPE p95={np.percentile(hpe,95):.1f}m",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.suptitle("Uçuş Trajektorisi: GPS Gerçeği vs GRU Kümülatif Tahmin", fontsize=13)
    plt.tight_layout()
    plt.savefig(PH4_PLOT / "trajectory_2d_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Plot: trajectory_2d_comparison.png")


# ── GPS Kesintisi Trajektorisi ──────────────────────────────────────────
def plot_outage_trajectory(npz_path, X_scaled, y_delta, y_abs, device, model,
                           outage_start_frac=0.40, outage_secs=30):
    N = len(X_scaled)
    outage_start = int(N * outage_start_frac)
    outage_end   = min(outage_start + int(outage_secs * FS), N)

    preds = run_inference(model, X_scaled, device)

    # GPS kesintisi öncesi: gerçek konum
    true_north = y_abs[:, 0]
    true_east  = y_abs[:, 1]

    # Kesinti sonrası kümülatif model tahmini
    start_pos = y_abs[outage_start, :2].copy()
    cum_north, cum_east = start_pos[0], start_pos[1]
    pred_n = [cum_north]; pred_e = [cum_east]
    for t in range(outage_start, outage_end):
        cum_north += preds[t, 0]
        cum_east  += preds[t, 1]
        pred_n.append(cum_north)
        pred_e.append(cum_east)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Tam gerçek rota
    ax.plot(true_east, true_north, "b-", lw=1.5, alpha=0.5, label="GPS Rotası (Gerçek)")
    # Kesinti öncesi
    ax.plot(true_east[:outage_start+1], true_north[:outage_start+1],
            "b-", lw=2.5, label="GPS Aktif")
    # Kesinti sonrası gerçek
    ax.plot(true_east[outage_start:outage_end+1], true_north[outage_start:outage_end+1],
            "g-", lw=2.5, label="GPS Kesintisi (Gerçek)")
    # Model tahmini
    ax.plot(pred_e, pred_n, "r--", lw=2.5, label="GRU Tahmini")

    # Kesinti noktası
    ax.axvline(x=true_east[outage_start], color="orange", ls=":", alpha=0.7)
    ax.plot(true_east[outage_start], true_north[outage_start], "o",
            ms=12, color="orange", zorder=5, label=f"Kesinti Başlangıcı (%{int(outage_start_frac*100)})")

    # Final hata
    final_hpe = np.sqrt((pred_n[-1]-true_north[outage_end])**2 +
                        (pred_e[-1]-true_east[outage_end])**2)
    ax.set(xlabel="East (m)", ylabel="North (m)",
           title=f"GPS Kesintisi Trajektorisi — %{int(outage_start_frac*100)} Başlangıç, {outage_secs}s Kesinti\n"
                 f"Final HPE = {final_hpe:.1f} m")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    name = npz_path.stem.replace("flight_", "").replace("_test", "")
    out = PH4_PLOT / f"outage_trajectory_{name}_f{int(outage_start_frac*100)}_{outage_secs}s.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out.name}")


# ── Irtifa Hatası ────────────────────────────────────────────────────────
def plot_altitude_error(flights_data, device, model):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (npz_path, X_scaled, y_delta, y_abs, _) in zip(axes, flights_data):
        preds = run_inference(model, X_scaled, device)
        # ΔUp: gerçek vs tahmin (per step)
        true_up = y_delta[:, 2]   # ground truth ΔUp
        pred_up = preds[:, 2]
        t_arr = np.arange(len(true_up)) * 0.5  # saniye
        ax.plot(t_arr, true_up, "b-", lw=1.2, alpha=0.7, label="Gerçek ΔUp")
        ax.plot(t_arr, pred_up, "r-", lw=1.2, alpha=0.7, label="Tahmin ΔUp")
        name = npz_path.stem.replace("flight_", "").replace("_test", "")
        ax.set(xlabel="Zaman (s)", ylabel="ΔUp (m)", title=name)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        rmse_up = float(np.sqrt(np.mean((pred_up - true_up)**2)))
        ax.text(0.02, 0.98, f"RMSE Up={rmse_up:.3f}m",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    plt.suptitle("İrtifa Değişimi (ΔUp): Gerçek vs GRU Tahmini", fontsize=13)
    plt.tight_layout()
    plt.savefig(PH4_PLOT / "altitude_delta_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Plot: altitude_delta_comparison.png")


# ── Sürüklenme Hızı (Drift Rate) ────────────────────────────────────────
def plot_drift_rate(npz_files, device, model):
    """Tüm test uçuşlarında ortalama drift hızı (m/s) vs uçuş süresi"""
    drift_rates = []
    flight_lens = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        X_scaled = data["X_scaled"].astype(np.float32)
        y_delta  = data["y_delta"].astype(np.float32)
        y_abs    = data["y_abs"].astype(np.float32)
        N = len(X_scaled)
        if N < 50:
            continue
        preds = run_inference(model, X_scaled, device)
        # Kümülatif tahmin başlangıçtan itibaren
        pos = reconstruct_trajectory(y_abs[0, :2], preds[:, :2])
        pred_pos = pos[1:, :]
        true_pos = y_abs[:, :2]
        hpe = np.sqrt(np.sum((pred_pos - true_pos)**2, axis=1))
        # Drift rate = son HPE / toplam zaman
        total_time = N * 0.5  # sn
        drift_rates.append(hpe[-1] / total_time)
        flight_lens.append(total_time)

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(flight_lens, drift_rates, c=drift_rates, cmap="YlOrRd", s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label="Drift Rate (m/s)")
    ax.set(xlabel="Uçuş Süresi (s)", ylabel="Drift Hızı (m/s)",
           title="GRU Kümülatif Drift Hızı — Her Test Uçuşu")
    ax.axhline(np.median(drift_rates), color="b", ls="--",
               label=f"Medyan={np.median(drift_rates):.3f} m/s")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PH4_PLOT / "drift_rate_per_flight.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot: drift_rate_per_flight.png  (medyan drift={np.median(drift_rates):.4f} m/s)")
    return drift_rates


# ── 3D Rota Grafiği ─────────────────────────────────────────────────────
def plot_3d_trajectory(flights_data, device, model):
    """2 örnek uçuş için 3D North-East-Alt trajektori"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(16, 7))
    for idx, (npz_path, X_scaled, y_delta, y_abs, _) in enumerate(flights_data[:2]):
        ax = fig.add_subplot(1, 2, idx+1, projection="3d")
        preds = run_inference(model, X_scaled, device)

        true_north = y_abs[:, 0]
        true_east  = y_abs[:, 1]
        true_alt   = y_abs[:, 2]   # Up direction

        pos = reconstruct_trajectory(y_abs[0, :3], preds[:, :3])
        pred_north = pos[1:, 0]
        pred_east  = pos[1:, 1]
        pred_alt   = pos[1:, 2]

        ax.plot(true_east, true_north, true_alt, "b-", lw=1.5, label="GPS (Gerçek)")
        ax.plot(pred_east, pred_north, pred_alt, "r--", lw=1.5, alpha=0.8, label="GRU Tahmin")
        ax.scatter([true_east[0]], [true_north[0]], [true_alt[0]], c="g", s=60, zorder=5)
        ax.scatter([true_east[-1]], [true_north[-1]], [true_alt[-1]], c="r", s=60, zorder=5)
        name = npz_path.stem.replace("flight_", "").replace("_test", "")
        ax.set(xlabel="East (m)", ylabel="North (m)", zlabel="Alt (m)", title=name)
        ax.legend(fontsize=8)

    plt.suptitle("3D Uçuş Trajektorisi: GPS Gerçeği vs GRU Tahmini", fontsize=13)
    plt.tight_layout()
    plt.savefig(PH4_PLOT / "trajectory_3d_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Plot: trajectory_3d_comparison.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"PHASE 4 — 01: Trajektori Grafikleri  [{device}]")
    print("=" * 60)

    model = load_gru(device)

    npz_files = sorted(PH1.glob("flight_*_test.npz"))
    print(f"  Test ucuslari: {len(npz_files)}")

    # İlk 4 uçuşu örnek olarak seç
    sample_flights = []
    for npz_path in npz_files[:4]:
        data = np.load(npz_path)
        sample_flights.append((
            npz_path,
            data["X_scaled"].astype(np.float32),
            data["y_delta"].astype(np.float32),
            data["y_abs"].astype(np.float32),
            data["time_s"].astype(np.float32),
        ))

    print("  Grafik 1: 2D trajektori karsilastirmasi...")
    plot_2d_trajectory(sample_flights, device, model)

    print("  Grafik 2: 3D trajektori karsilastirmasi...")
    plot_3d_trajectory(sample_flights, device, model)

    print("  Grafik 4: GPS kesintisi trajektori (30s @ %40)...")
    npz_path, X_scaled, y_delta, y_abs, ts = sample_flights[0]
    plot_outage_trajectory(npz_path, X_scaled, y_delta, y_abs, device, model,
                           outage_start_frac=0.40, outage_secs=30)

    print("  Grafik 5: GPS kesintisi trajektori (60s @ %20)...")
    plot_outage_trajectory(npz_path, X_scaled, y_delta, y_abs, device, model,
                           outage_start_frac=0.20, outage_secs=60)

    print("  Grafik 4: İrtifa deltası karsilastirmasi...")
    plot_altitude_error(sample_flights, device, model)

    print("  Grafik 5: Drift hizi analizi...")
    drift_rates = plot_drift_rate(npz_files, device, model)
    print(f"    Drift rate: min={min(drift_rates):.4f}  max={max(drift_rates):.4f}  "
          f"median={float(np.median(drift_rates)):.4f} m/s")

    print("\nAdim 1 tamamlandi.")


if __name__ == "__main__":
    main()
