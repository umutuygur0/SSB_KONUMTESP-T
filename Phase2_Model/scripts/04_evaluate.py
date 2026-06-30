"""
Phase 2 — Adım 4: Kapsamlı Değerlendirme
  1. Batch inference  → metrics_gru.json / metrics_lstm.json / metrics_baseline.json güncelleme
  2. GPS kesinti deneyi (10s / 30s / 60s, başlangıç %20 / %40 / %60)
  3. Grafikler:
     • HPE kutu grafiği (GRU vs LSTM vs DR)
     • Örnek uçuş rotası 2D (gerçek vs tahmin)
     • Yatay hata vs zaman (outage sırasında)
     • GPS kesinti süresi vs ortalama HPE
"""

import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import GRUModel, LSTMModel, detrend_windows, detrend_single

BASE_DIR   = Path(__file__).resolve().parents[2]
PHASE1_OUT = BASE_DIR / "Phase1_Veri_Analizi" / "outputs"
PHASE2_OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS_DIR  = PHASE2_OUT / "plots"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 40      # pencere uzunluğu (adım)
DT       = 0.5     # saniye / adım (2 Hz)
DV_IDX   = slice(3, 6)

# Kesinti süreleri (adım cinsinden)
OUTAGE_DURATIONS_S  = [10, 30, 60]             # saniye
OUTAGE_DURATIONS_ST = [int(s / DT) for s in OUTAGE_DURATIONS_S]  # adım
OUTAGE_FRACS        = [0.20, 0.40, 0.60]       # başlangıç konumu


# ── Yardımcı metrik fonksiyonu ──────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err  = y_true - y_pred
    dn, de, du = err[:, 0], err[:, 1], err[:, 2]
    hpe  = np.sqrt(dn**2 + de**2)
    e3d  = np.sqrt(dn**2 + de**2 + du**2)
    rmse = np.sqrt((err**2).mean(axis=0))
    mae  = np.abs(err).mean(axis=0)
    return {
        "HPE_mean":   float(hpe.mean()),
        "HPE_median": float(np.median(hpe)),
        "HPE_p95":    float(np.percentile(hpe, 95)),
        "3DE_mean":   float(e3d.mean()),
        "3DE_median": float(np.median(e3d)),
        "RMSE_north": float(rmse[0]),
        "RMSE_east":  float(rmse[1]),
        "RMSE_up":    float(rmse[2]),
        "RMSE_3D":    float(np.sqrt((err**2).mean())),
        "MAE_north":  float(mae[0]),
        "MAE_east":   float(mae[1]),
        "MAE_up":     float(mae[2]),
        "n_samples":  int(len(y_true)),
    }


# ── Model yükleme ───────────────────────────────────────────────────────────

def load_gru() -> GRUModel:
    m = GRUModel().to(DEVICE)
    m.load_state_dict(torch.load(PHASE2_OUT / "best_gru.pt", map_location=DEVICE))
    m.eval()
    return m


def load_lstm() -> LSTMModel:
    m = LSTMModel().to(DEVICE)
    m.load_state_dict(torch.load(PHASE2_OUT / "best_lstm.pt", map_location=DEVICE))
    m.eval()
    return m


# ── Batch inference ─────────────────────────────────────────────────────────

def batch_predict(model, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    """X: (N, 40, 12) → preds: (N, 3)"""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            Xb = torch.FloatTensor(X[i:i + batch]).to(DEVICE)
            preds.append(model(Xb).cpu().numpy())
    return np.vstack(preds)


def dead_reckoning_predict(X_scaled: np.ndarray, scaler) -> np.ndarray:
    X_last_raw = scaler.inverse_transform(X_scaled[:, -1, :])
    dv = X_last_raw[:, DV_IDX]
    return np.column_stack([dv[:, 0] * DT, dv[:, 1] * DT, -dv[:, 2] * DT])


# ── GPS kesinti simülasyonu (tek uçuş) ─────────────────────────────────────

def simulate_outage(
    model,
    X_scaled: np.ndarray,   # (T, 12) — ham ölçeklenmiş, pencere olmayan
    y_abs: np.ndarray,      # (T, 3)  — [x_f58(N), y_f58(E), z_f58(Down)]
    outage_start: int,
    outage_duration: int,   # adım
) -> list[float]:
    """
    outage_start anından itibaren `outage_duration` adım boyunca
    GPS olmadan konum tahmin eder.
    Dönüş: her adımdaki HPE listesi (metre).
    """
    n = len(X_scaled)
    if outage_start < SEQ_LEN or outage_start >= n:
        return []

    pos_est = y_abs[outage_start].copy()   # [North, East, Down]
    hpe_list = []

    for step in range(1, outage_duration + 1):
        t = outage_start + step
        if t >= n:
            break

        # window = X_scaled[t-39 : t+1]  →  model window ending at t
        w_start = t - SEQ_LEN + 1
        if w_start < 0:
            pad = np.zeros((-w_start, X_scaled.shape[1]), dtype=np.float32)
            window = np.vstack([pad, X_scaled[:t + 1]])
        else:
            window = X_scaled[w_start:t + 1]

        window = detrend_single(window)   # mag+baro DC removal

        with torch.no_grad():
            xt = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
            pred = model(xt).cpu().numpy()[0]   # [ΔN, ΔE, ΔUp]

        pos_est[0] += pred[0]   # North  += ΔNorth
        pos_est[1] += pred[1]   # East   += ΔEast
        pos_est[2] -= pred[2]   # z_Down -= ΔUp  (ΔUp = -Δz_f58)

        true_pos = y_abs[t]
        hpe = np.sqrt((pos_est[0] - true_pos[0])**2 +
                      (pos_est[1] - true_pos[1])**2)
        hpe_list.append(float(hpe))

    return hpe_list


def simulate_outage_dr(
    X_scaled: np.ndarray,
    scaler,
    y_abs: np.ndarray,
    outage_start: int,
    outage_duration: int,
) -> list[float]:
    """Dead reckoning ile GPS kesinti simülasyonu."""
    n = len(X_scaled)
    if outage_start < SEQ_LEN or outage_start >= n:
        return []

    pos_est = y_abs[outage_start].copy()
    hpe_list = []

    for step in range(1, outage_duration + 1):
        t = outage_start + step
        if t >= n:
            break

        # DR: delta_velocity'yi ters-normalize et (mag/baro detrend gerekmez)
        raw = scaler.inverse_transform(X_scaled[t:t + 1])   # (1, 12)
        dv  = raw[0, DV_IDX]
        pos_est[0] += dv[0] * DT
        pos_est[1] += dv[1] * DT
        pos_est[2] -= (-dv[2] * DT)   # z_Down -= ΔUp

        true_pos = y_abs[t]
        hpe = np.sqrt((pos_est[0] - true_pos[0])**2 +
                      (pos_est[1] - true_pos[1])**2)
        hpe_list.append(float(hpe))

    return hpe_list


# ── Çizim fonksiyonları ─────────────────────────────────────────────────────

def plot_hpe_boxplot(hpe_dict: dict[str, np.ndarray]):
    """GRU / LSTM / Dead Reckoning HPE kutu grafiği."""
    labels = list(hpe_dict.keys())
    data   = [hpe_dict[k] for k in labels]
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    showfliers=False, whis=[5, 95])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("HPE (m)", fontsize=12)
    ax.set_title("Yatay Konum Hatası — Model Karşılaştırması\n(Whisker: P5–P95)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Ortalamaları yazdır
    for i, (lbl, d) in enumerate(zip(labels, data), 1):
        ax.text(i, np.median(d) + 0.3, f"med={np.median(d):.2f}m",
                ha="center", va="bottom", fontsize=9, color="black")

    plt.tight_layout()
    out = PLOTS_DIR / "hpe_comparison_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_sample_trajectory(
    gru_model, lstm_model, scaler,
    X_scaled: np.ndarray,   # (T, 12)
    y_abs: np.ndarray,      # (T, 3)
    flight_id: str,
):
    """Tek uçuşta 2D rota: GPS vs GRU vs LSTM vs Dead Reckoning."""
    T = len(X_scaled)
    if T < SEQ_LEN + 10:
        return

    # Her adım için model tahmini (window sliding, step=1)
    def traj_from_model(model):
        pos = y_abs[SEQ_LEN - 1].copy()
        north_list = [pos[0]]
        east_list  = [pos[1]]
        for t in range(SEQ_LEN, T):
            window = detrend_single(X_scaled[t - SEQ_LEN + 1:t + 1])
            with torch.no_grad():
                pred = model(torch.FloatTensor(window).unsqueeze(0).to(DEVICE)
                             ).cpu().numpy()[0]
            pos[0] += pred[0]; pos[1] += pred[1]; pos[2] -= pred[2]
            north_list.append(pos[0])
            east_list.append(pos[1])
        return np.array(north_list), np.array(east_list)

    def traj_dr():
        pos = y_abs[SEQ_LEN - 1].copy()
        north_list = [pos[0]]
        east_list  = [pos[1]]
        for t in range(SEQ_LEN, T):
            raw = scaler.inverse_transform(X_scaled[t:t + 1])[0]
            dv = raw[DV_IDX]
            pos[0] += dv[0] * DT; pos[1] += dv[1] * DT
            north_list.append(pos[0])
            east_list.append(pos[1])
        return np.array(north_list), np.array(east_list)

    steps = range(SEQ_LEN, T + 1)
    gru_n,  gru_e  = traj_from_model(gru_model)
    lstm_n, lstm_e = traj_from_model(lstm_model)
    dr_n,   dr_e   = traj_dr()
    true_n = y_abs[SEQ_LEN - 1:, 0]
    true_e = y_abs[SEQ_LEN - 1:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(true_e, true_n,  "k-",  lw=2,   label="GPS (Gerçek)", zorder=5)
    ax.plot(gru_e,  gru_n,   "b--", lw=1.5, label="GRU Tahmini",  alpha=0.85)
    ax.plot(lstm_e, lstm_n,  "r:",  lw=1.5, label="LSTM Tahmini", alpha=0.85)
    ax.plot(dr_e,   dr_n,    "g-.", lw=1.2, label="Dead Reckoning", alpha=0.7)
    ax.scatter([true_e[0]],  [true_n[0]],  s=80, c="k",       marker="o", zorder=6, label="Başlangıç")
    ax.scatter([true_e[-1]], [true_n[-1]], s=80, c="crimson",  marker="X", zorder=6, label="Bitiş")

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title(f"Uçuş Rotası — {flight_id}\n(Kümülatif pozisyon tahmini, Teacher Forcing YOK)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "sample_trajectory_2d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_hpe_vs_time(
    gru_hpe_t: np.ndarray,
    lstm_hpe_t: np.ndarray,
    dr_hpe_t: np.ndarray,
):
    """Ortalama HPE (m) vs kesinti süresi (s) — tüm uçuşlar ortalaması."""
    max_steps = max(len(gru_hpe_t), len(lstm_hpe_t), len(dr_hpe_t))
    time_ax   = np.arange(1, max_steps + 1) * DT

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_ax[:len(gru_hpe_t)],  gru_hpe_t,  label="GRU",            color="steelblue",  lw=2)
    ax.plot(time_ax[:len(lstm_hpe_t)], lstm_hpe_t, label="LSTM",           color="darkorange",  lw=2)
    ax.plot(time_ax[:len(dr_hpe_t)],   dr_hpe_t,   label="Dead Reckoning", color="seagreen",    lw=2, ls="--")
    ax.set_xlabel("GPS Kesinti Süresi (s)", fontsize=12)
    ax.set_ylabel("Ortalama HPE (m)", fontsize=12)
    ax.set_title("Yatay Hata vs GPS Kesinti Süresi\n(tüm test uçuşları & başlangıç noktaları ortalaması)", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "hpe_vs_outage_time.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_outage_duration_bar(results: dict):
    """
    results[model][duration_s] = mean_final_hpe
    Bar chart: duration vs mean_HPE, grouped by model.
    """
    models   = list(results.keys())
    dur_list = sorted({d for m in results for d in results[m]})
    x        = np.arange(len(dur_list))
    width    = 0.25
    colors   = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model].get(d, 0.0) for d in dur_list]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{d}s" for d in dur_list], fontsize=12)
    ax.set_xlabel("GPS Kesinti Süresi", fontsize=12)
    ax.set_ylabel("Ortalama Bitiş HPE (m)", fontsize=12)
    ax.set_title("GPS Kesinti Süresi vs Ortalama HPE\n(tüm test uçuşları & başlangıç noktaları ortalaması)", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOTS_DIR / "gps_outage_duration_vs_hpe.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


# ── Ana fonksiyon ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ADIM 4: Değerlendirme & GPS Kesinti Deneyi")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Veri & model yükleme ──────────────────────────────────────────
    X_test = detrend_windows(np.load(PHASE1_OUT / "X_test.npy"))
    y_test = np.load(PHASE1_OUT / "y_test.npy")
    with open(PHASE1_OUT / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"\n[1] Batch inference  —  X_test: {X_test.shape}")

    gru_model  = load_gru()
    lstm_model = load_lstm()

    gru_preds  = batch_predict(gru_model,  X_test)
    lstm_preds = batch_predict(lstm_model, X_test)
    dr_preds   = dead_reckoning_predict(X_test, scaler)

    m_gru  = compute_metrics(y_test, gru_preds)
    m_lstm = compute_metrics(y_test, lstm_preds)
    m_dr   = compute_metrics(y_test, dr_preds)

    for tag, m in [("GRU", m_gru), ("LSTM", m_lstm), ("DR", m_dr)]:
        print(f"\n  [{tag}]")
        print(f"    HPE  mean / median / P95 : {m['HPE_mean']:.3f} / {m['HPE_median']:.3f} / {m['HPE_p95']:.3f} m")
        print(f"    3DE  mean / median       : {m['3DE_mean']:.3f} / {m['3DE_median']:.3f} m")
        print(f"    RMSE (N, E, Up, 3D)      : {m['RMSE_north']:.3f} / {m['RMSE_east']:.3f} / {m['RMSE_up']:.3f} / {m['RMSE_3D']:.3f} m")

    with open(PHASE2_OUT / "metrics_gru.json",      "w", encoding="utf-8") as f:
        json.dump(m_gru,  f, indent=2)
    with open(PHASE2_OUT / "metrics_lstm.json",     "w", encoding="utf-8") as f:
        json.dump(m_lstm, f, indent=2)
    with open(PHASE2_OUT / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(m_dr,   f, indent=2)
    print("\n  Metrikler kaydedildi.")

    # ── HPE kutu grafiği ──────────────────────────────────────────────
    print("\n[2] HPE kutu grafiği")
    hpe_gru  = np.sqrt(((y_test - gru_preds)[:, 0])**2 +
                       ((y_test - gru_preds)[:, 1])**2)
    hpe_lstm = np.sqrt(((y_test - lstm_preds)[:, 0])**2 +
                       ((y_test - lstm_preds)[:, 1])**2)
    hpe_dr   = np.sqrt(((y_test - dr_preds)[:, 0])**2 +
                       ((y_test - dr_preds)[:, 1])**2)
    plot_hpe_boxplot({"GRU": hpe_gru, "LSTM": hpe_lstm, "Dead Reckoning": hpe_dr})

    # ── Test uçuşlarını yükle ─────────────────────────────────────────
    print("\n[3] Test uçuşları yükleniyor")
    test_npz_files = sorted(PHASE1_OUT.glob("flight_*_test.npz"))
    print(f"  {len(test_npz_files)} test uçuşu bulundu")

    flights = []
    for fp in test_npz_files:
        d = np.load(fp, allow_pickle=True)
        X_sc   = d["X_scaled"]
        y_ab   = d["y_abs"]
        y_dl   = d["y_delta"]
        if len(X_sc) < SEQ_LEN + max(OUTAGE_DURATIONS_ST) + 5:
            continue
        flights.append({
            "id":       fp.stem,
            "X_scaled": X_sc,
            "y_abs":    y_ab,
            "y_delta":  y_dl,
        })
    print(f"  Kullanılabilir: {len(flights)} uçuş")

    # ── Örnek uçuş rotası ─────────────────────────────────────────────
    if flights:
        print("\n[4] Örnek uçuş rotası çiziliyor")
        fl = flights[0]
        plot_sample_trajectory(
            gru_model, lstm_model, scaler,
            fl["X_scaled"], fl["y_abs"], fl["id"],
        )

    # ── GPS kesinti deneyi ────────────────────────────────────────────
    print("\n[5] GPS kesinti deneyi")
    MAX_DUR = max(OUTAGE_DURATIONS_ST)

    # Her model için: [flight x frac x step] → HPE zaman serisi (max dur'a kadar)
    gru_hpe_matrix  = []
    lstm_hpe_matrix = []
    dr_hpe_matrix   = []

    outage_results = {
        "GRU":            {d: [] for d in OUTAGE_DURATIONS_S},
        "LSTM":           {d: [] for d in OUTAGE_DURATIONS_S},
        "Dead Reckoning": {d: [] for d in OUTAGE_DURATIONS_S},
    }

    for fi, fl in enumerate(flights):
        X_sc = fl["X_scaled"]
        y_ab = fl["y_abs"]
        T    = len(X_sc)

        for frac in OUTAGE_FRACS:
            s = int(T * frac)
            if s < SEQ_LEN or s + MAX_DUR >= T:
                continue

            g_hpe  = simulate_outage(gru_model,  X_sc, y_ab, s, MAX_DUR)
            l_hpe  = simulate_outage(lstm_model, X_sc, y_ab, s, MAX_DUR)
            dr_hpe = simulate_outage_dr(X_sc, scaler, y_ab, s, MAX_DUR)

            if not g_hpe:
                continue

            gru_hpe_matrix.append(g_hpe)
            lstm_hpe_matrix.append(l_hpe)
            dr_hpe_matrix.append(dr_hpe)

            # Belli sürelerdeki son HPE
            for dur_s, dur_st in zip(OUTAGE_DURATIONS_S, OUTAGE_DURATIONS_ST):
                idx = dur_st - 1
                if idx < len(g_hpe):
                    outage_results["GRU"][dur_s].append(g_hpe[idx])
                if idx < len(l_hpe):
                    outage_results["LSTM"][dur_s].append(l_hpe[idx])
                if idx < len(dr_hpe):
                    outage_results["Dead Reckoning"][dur_s].append(dr_hpe[idx])

        if (fi + 1) % 5 == 0:
            print(f"  {fi + 1}/{len(flights)} uçuş işlendi")

    print(f"  Toplam kombinasyon: {len(gru_hpe_matrix)}")

    # Ortalama HPE zaman serisi
    def mean_ts(matrix, max_len):
        if not matrix:
            return np.zeros(max_len)
        arr = np.zeros((len(matrix), max_len))
        for i, row in enumerate(matrix):
            n = min(len(row), max_len)
            arr[i, :n] = row[:n]
            if n < max_len:   # son değeri taşı
                arr[i, n:] = row[-1] if row else 0
        return arr.mean(axis=0)

    gru_ts  = mean_ts(gru_hpe_matrix,  MAX_DUR)
    lstm_ts = mean_ts(lstm_hpe_matrix, MAX_DUR)
    dr_ts   = mean_ts(dr_hpe_matrix,   MAX_DUR)

    print("\n[6] HPE vs zaman grafiği")
    plot_hpe_vs_time(gru_ts, lstm_ts, dr_ts)

    # Ortalama final HPE bar chart
    bar_data = {
        model: {d: float(np.mean(vals)) if vals else 0.0
                for d, vals in outage_results[model].items()}
        for model in outage_results
    }
    print("\n[7] GPS kesinti süresi vs HPE bar grafiği")
    plot_outage_duration_bar(bar_data)

    # Kesinti sonuçları JSON
    outage_summary = {
        model: {
            str(d) + "s": {
                "mean_hpe_m": float(np.mean(vals)) if vals else None,
                "median_hpe_m": float(np.median(vals)) if vals else None,
                "n": len(vals),
            }
            for d, vals in outage_results[model].items()
        }
        for model in outage_results
    }
    with open(PHASE2_OUT / "outage_results.json", "w", encoding="utf-8") as f:
        json.dump(outage_summary, f, indent=2)
    print("\n  Kesinti sonuçları: outage_results.json")

    # Kesinti tablosu konsola yazdır
    print("\n  GPS Kesinti Deneyi — Ortalama Final HPE (m):")
    print(f"  {'Süre':>8}  {'GRU':>10}  {'LSTM':>10}  {'Dead Rec.':>12}")
    for dur_s in OUTAGE_DURATIONS_S:
        g  = bar_data["GRU"].get(dur_s, 0)
        l  = bar_data["LSTM"].get(dur_s, 0)
        dr = bar_data["Dead Reckoning"].get(dur_s, 0)
        print(f"  {dur_s:>6}s   {g:>10.2f}  {l:>10.2f}  {dr:>12.2f}")

    print("\nAdım 4 tamamlandı.\n")
    return m_gru, m_lstm, m_dr, outage_summary


if __name__ == "__main__":
    main()
