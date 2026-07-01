"""
Phase 3 — GPS Kesintisi Deneyi (Tam Trajektori)

Protokol:
  - Test NPZ dosyalarından her uçuş yüklenir
  - Kesinti başlangıcı: uçuşun %20, %40, %60'ında
  - Kesinti süresi: 10s, 30s, 60s ve "kalan tümü"
  - Teacher forcing YOK — model çıktısı kümülatif olarak toplanır
  - HPE = sqrt(ΔN² + ΔE²) hem GRU, LSTM, Ensemble hem Dead Reckoning için
  - Grafikler: HPE vs zaman, matris ısı haritası, rota karşılaştırması
"""

import numpy as np
import torch
import torch.nn as nn
import json
import importlib.util
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE     = Path(__file__).resolve().parents[2]
PH1      = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2      = BASE / "Phase2_Model" / "outputs"
PH3      = Path(__file__).resolve().parents[1] / "outputs"
PLOT_DIR = PH3 / "plots"
SCRIPTS2 = BASE / "Phase2_Model" / "scripts"

# models.py yükleme
_spec = importlib.util.spec_from_file_location("models", SCRIPTS2 / "models.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
detrend_windows = _m.detrend_windows
GRUModel  = _m.GRUModel
LSTMModel = _m.LSTMModel

# ─── Sabitler ────────────────────────────────────────────────────────────
SEQ_LEN    = 40          # 20s pencere
STEP       = 4           # 2s adım
FS         = 2           # Hz (0.5s/örnek)
DT         = 1.0 / FS   # 0.5s

START_FRACS   = [0.20, 0.40, 0.60]    # kesinti başlangıç oranları
OUTAGE_SECS   = [10, 30, 60, None]    # kesinti süreleri (None = tüm kalan)
OUTAGE_LABELS = ["10s", "30s", "60s", "Tum"]
ENS_W_GRU  = 0.75   # GRU ağırlığı (0.5 eşit, 0.75 optimal)
ENS_W_LSTM = 0.25


# ─── Model yükleme ────────────────────────────────────────────────────────
def load_models(device):
    gru = GRUModel().to(device)
    gru.load_state_dict(torch.load(PH2 / "best_gru.pt",  weights_only=True))
    gru.eval()
    lstm = LSTMModel().to(device)
    lstm.load_state_dict(torch.load(PH2 / "best_lstm.pt", weights_only=True))
    lstm.eval()
    return gru, lstm


# ─── Per-pencere inference ────────────────────────────────────────────────
@torch.no_grad()
def predict_window(model, window_np, device):
    """window_np: (40, 12) raw-scaled — detrend yapılır, (3,) delta tahmin döner."""
    x = window_np[None, :, :].copy()  # (1,40,12)
    # detrend (mag + baro)
    LOCAL = [6, 7, 8, 10]
    for c in LOCAL:
        col = x[0, :, c]
        mu = col.mean(); sig = col.std()
        x[0, :, c] = (col - mu) / (sig + 1e-8)
    t = torch.from_numpy(x.astype(np.float32)).to(device)
    return model(t).cpu().numpy()[0]  # (3,)


def predict_window_ensemble(gru, lstm, window_np, device):
    pg = predict_window(gru,  window_np, device)
    pl = predict_window(lstm, window_np, device)
    return ENS_W_GRU * pg + ENS_W_LSTM * pl


def dead_reckoning_window(window_np):
    """Son pencerede ortalama delta_velocity * DT — basit fiziksel hesap."""
    # cols: 3-5 delta_velocity[0-2]
    dv = window_np[:, 3:6].mean(axis=0)   # (3,)
    return np.array([dv[0]*DT, dv[1]*DT, -dv[2]*DT], dtype=np.float32)


# ─── Tek uçuş simülasyonu ────────────────────────────────────────────────
def simulate_flight(X_raw, y_true, gru, lstm, device,
                    outage_start_step, outage_end_step):
    """
    X_raw: (N, 40, 12) — ham-scaled pencereler (detrend yapılmadı)
    y_true: (N, 3)     — gerçek delta [m]
    outage_start_step: kesintinin başladığı pencere indexi
    outage_end_step:   kesintinin bittiği pencere indexi (exclusive)

    Döndürür: dict of arrays, her model için HPE[t] eğrisi
    """
    N = len(X_raw)
    end = min(outage_end_step, N)

    # Kümülatif konum hatası başlangıç noktası = 0
    # Model yalnızca [outage_start, end) aralığında çalışır
    # Bu aralıkta gerçek delta yerine model tahmini kullanılır

    results = {}
    for label, fn in [
        ("GRU",      lambda w: predict_window(gru, w, device)),
        ("LSTM",     lambda w: predict_window(lstm, w, device)),
        ("Ensemble", lambda w: predict_window_ensemble(gru, lstm, w, device)),
        ("DR",       lambda w: dead_reckoning_window(w)),
    ]:
        cum_north = 0.0; cum_east = 0.0
        hpe_series = []
        for t in range(outage_start_step, end):
            pred = fn(X_raw[t])
            true = y_true[t]
            # Ne kadar sapıyoruz?
            cum_north += (pred[0] - true[0])
            cum_east  += (pred[1] - true[1])
            hpe_series.append(np.sqrt(cum_north**2 + cum_east**2))
        results[label] = np.array(hpe_series, dtype=np.float32)
    return results


# ─── Ana simülasyon ──────────────────────────────────────────────────────
def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Cihaz: {device}")
    gru, lstm = load_models(device)

    # Test NPZ'lerini yükle — her dosya bir uçuş
    npz_files = sorted(PH1.glob("flight_*_test.npz"))
    print(f"  Test ucuslari: {len(npz_files)}")

    # Sonuç matrisi: [start_frac][outage_dur] = list of final HPE per flight
    # keys: ("GRU", 0.20, "10s"), ("LSTM", ...), ("Ensemble", ...), ("DR", ...)
    from collections import defaultdict
    matrix = defaultdict(list)   # (model, frac, dur) -> [float]

    hpe_curves = defaultdict(list)  # (frac, dur) -> list of arrays (for avg curve)

    for npz_path in npz_files:
        data = np.load(npz_path)
        # NPZ: X_scaled=(N,12) per-row, y_delta=(N,3) per-row
        X_scaled = data["X_scaled"].astype(np.float32)  # (N, 12)
        y        = data["y_delta"].astype(np.float32)   # (N, 3)
        N = len(X_scaled)
        if N < SEQ_LEN + 20:
            continue

        # Sliding window oluştur: her satır t için X[t-SEQ_LEN+1:t+1]
        # Sadece t >= SEQ_LEN-1 için geçerli pencere
        def make_window(X_s, t):
            start_w = max(0, t - SEQ_LEN + 1)
            w = X_s[start_w:t+1, :]          # (<=SEQ_LEN, 12)
            if len(w) < SEQ_LEN:
                # ön tarafı sıfır doldur
                pad = np.zeros((SEQ_LEN - len(w), 12), dtype=np.float32)
                w = np.concatenate([pad, w], axis=0)
            return w   # (SEQ_LEN, 12)

        # X_raw olarak per-row sliding-window dizisi kullan
        X_raw = np.stack([make_window(X_scaled, t) for t in range(N)], axis=0)  # (N,40,12)

        for frac in START_FRACS:
            start = int(N * frac)
            if start >= N - 5:
                continue

            for dur_s, dur_lbl in zip(OUTAGE_SECS, OUTAGE_LABELS):
                if dur_s is None:
                    end = N
                else:
                    dur_steps = int(dur_s * FS)  # sn → satır (2Hz × sn)
                    end = min(start + dur_steps, N)
                if end <= start:
                    continue

                sim = simulate_flight(X_raw, y, gru, lstm, device, start, end)
                key = (frac, dur_lbl)
                for model_name, hpe_arr in sim.items():
                    if len(hpe_arr) > 0:
                        matrix[(model_name, frac, dur_lbl)].append(float(hpe_arr[-1]))
                        hpe_curves[(model_name, frac, dur_lbl)].append(hpe_arr)

    # Ortalama final HPE tablosu
    summary = {}
    for (model_name, frac, dur_lbl), vals in matrix.items():
        key = f"{model_name}|{frac}|{dur_lbl}"
        summary[key] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "n_flights": len(vals),
        }

    with open(PH3 / "outage_matrix_v3.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, hpe_curves, matrix


# ─── Raporlama / Grafikler ───────────────────────────────────────────────
def print_table(summary):
    models = ["GRU", "LSTM", "Ensemble", "DR"]
    print(f"\n  {'Model':10s} | {'Basl%':6s} | {'Sure':5s} | {'Ort HPE':9s} | {'P95 HPE':9s} | N")
    print("  " + "-" * 60)
    for frac in START_FRACS:
        for dur_lbl in OUTAGE_LABELS:
            for model in models:
                k = f"{model}|{frac}|{dur_lbl}"
                v = summary.get(k, {})
                if v:
                    print(f"  {model:10s} | {frac*100:5.0f}% | {dur_lbl:5s} | "
                          f"{v['mean']:9.2f}m | {v['p95']:9.2f}m | {v['n_flights']}")


def plot_heatmap(summary, models=("GRU", "LSTM", "Ensemble")):
    """Model başına 3x4 ısı haritası — frac×dur"""
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    for ax, model in zip(axes, models):
        data = np.zeros((len(START_FRACS), len(OUTAGE_LABELS)))
        for i, frac in enumerate(START_FRACS):
            for j, dur in enumerate(OUTAGE_LABELS):
                k = f"{model}|{frac}|{dur}"
                v = summary.get(k, {})
                data[i, j] = v.get("mean", np.nan)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(OUTAGE_LABELS))); ax.set_xticklabels(OUTAGE_LABELS)
        ax.set_yticks(range(len(START_FRACS)));   ax.set_yticklabels([f"%{int(f*100)}" for f in START_FRACS])
        ax.set_title(f"{model} — Ort. HPE (m)")
        ax.set_xlabel("Kesinti Suresi"); ax.set_ylabel("Kesinti Baslangic")
        plt.colorbar(im, ax=ax)
        for i in range(len(START_FRACS)):
            for j in range(len(OUTAGE_LABELS)):
                ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color="black" if data[i,j] < data.max()*0.6 else "white")
    plt.suptitle("GPS Kesinti Deneyi — Ortalama Final HPE (metre)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "outage_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Plot: outage_heatmap.png")


def plot_hpe_vs_time(hpe_curves, frac=0.40, dur="30s"):
    """Seçili senaryo için ortalama HPE eğrisi"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"GRU": "steelblue", "LSTM": "coral", "Ensemble": "green", "DR": "gray"}
    for model in ["GRU", "LSTM", "Ensemble", "DR"]:
        k = (model, frac, dur)
        curves = hpe_curves.get(k, [])
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.array([np.pad(c, (0, max_len-len(c)), constant_values=c[-1] if len(c)>0 else 0) for c in curves])
        mean_c = padded.mean(axis=0)
        p25    = np.percentile(padded, 25, axis=0)
        p75    = np.percentile(padded, 75, axis=0)
        t = (np.arange(len(mean_c)) + 1) * STEP / FS  # saniye
        ax.plot(t, mean_c, label=model, color=colors.get(model, "k"), lw=2)
        ax.fill_between(t, p25, p75, alpha=0.15, color=colors.get(model, "k"))
    ax.set(xlabel="Kesinti Sonrasi Sure (s)", ylabel="HPE (m)",
           title=f"Ortalama Yatay Konum Hatasi — Baslangic=%{int(frac*100)}, Sure={dur}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"hpe_vs_time_f{int(frac*100)}_{dur}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot: hpe_vs_time_f{int(frac*100)}_{dur}.png")


def plot_bar_comparison(summary, frac=0.40):
    """Tek başlangıç fraksiyonu için 4 model × 4 süre bar grafiği"""
    models = ["GRU", "LSTM", "Ensemble", "DR"]
    x = np.arange(len(OUTAGE_LABELS))
    width = 0.2
    colors = ["steelblue", "coral", "green", "gray"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [summary.get(f"{model}|{frac}|{dur}", {}).get("mean", 0) for dur in OUTAGE_LABELS]
        ax.bar(x + i*width, vals, width, label=model, color=color, alpha=0.85)
    ax.set_xticks(x + 1.5*width); ax.set_xticklabels(OUTAGE_LABELS)
    ax.set(xlabel="Kesinti Suresi", ylabel="Ortalama Final HPE (m)",
           title=f"Model Karsilastirmasi — Kesinti Baslangici %{int(frac*100)}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"bar_comparison_f{int(frac*100)}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot: bar_comparison_f{int(frac*100)}.png")


def plot_trajectory_sample(hpe_curves, matrix, frac=0.40, dur="60s"):
    """Örnek uçuş HPE eğrisi (ilk 3 uçuş)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"GRU": "steelblue", "LSTM": "coral", "Ensemble": "green", "DR": "gray"}
    for model in ["GRU", "LSTM", "Ensemble", "DR"]:
        k = (model, frac, dur)
        curves = hpe_curves.get(k, [])
        if len(curves) == 0:
            continue
        c = curves[0]   # sadece ilk uçuş
        t = (np.arange(len(c)) + 1) * STEP / FS
        ax.plot(t, c, label=model, color=colors.get(model, "k"), lw=2)
    ax.set(xlabel="Kesinti Sonrasi Sure (s)", ylabel="HPE (m)",
           title=f"Ornek Ucus HPE — Baslangic=%{int(frac*100)}, Sure={dur}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"sample_ucus_hpe_f{int(frac*100)}_{dur}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot: sample_ucus_hpe_f{int(frac*100)}_{dur}.png")


def main():
    print("=" * 60)
    print("PHASE 3 — GPS Kesintisi Tam Trajektori Deneyi")
    print("=" * 60)

    summary, hpe_curves, matrix = run_all()
    print_table(summary)

    print("\n  Grafikler olusturuluyor...")
    plot_heatmap(summary)
    for frac in START_FRACS:
        for dur in ["30s", "60s", "Tum"]:
            plot_hpe_vs_time(hpe_curves, frac=frac, dur=dur)
    for frac in START_FRACS:
        plot_bar_comparison(summary, frac=frac)
    plot_trajectory_sample(hpe_curves, matrix, frac=0.40, dur="60s")

    print("\n  Ozet (frac=40%, sure=60s):")
    for model in ["GRU", "LSTM", "Ensemble", "DR"]:
        k = f"{model}|0.4|60s"
        v = summary.get(k, {})
        print(f"    {model:10s}: mean={v.get('mean',0):.1f}m  p95={v.get('p95',0):.1f}m")

    print(f"\nTamamlandi. Cikti: {PH3}")


if __name__ == "__main__":
    main()
