"""
Phase 4 – 02: GPS Kesintisi Boxplot & Drift Rate Analizi
  - Farklı kesinti noktaları için HPE dağılımı (boxplot)
  - Model başına drift hızı karşılaştırması
  - Phase 2 + Phase 3 kombine özet
"""

import numpy as np
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = Path(__file__).resolve().parents[2]
PH3_OUT  = BASE / "Phase3_GPS_Kesintisi" / "outputs"
PH2_OUT  = BASE / "Phase2_Model" / "outputs"
PH4_PLOT = Path(__file__).resolve().parents[1] / "outputs" / "plots"


def load_matrix():
    with open(PH3_OUT / "outage_matrix_v3.json") as f:
        return json.load(f)


def load_ph3_raw():
    """Phase 3 simülasyon verisi — her uçuş için final HPE listelerini yeniden oluştur."""
    import numpy as np, torch, importlib.util
    from pathlib import Path

    SCRIPTS2 = BASE / "Phase2_Model" / "scripts"
    PH1 = BASE / "Phase1_Veri_Analizi" / "outputs"

    _spec = importlib.util.spec_from_file_location("models", SCRIPTS2 / "models.py")
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    GRUModel = _m.GRUModel; LSTMModel = _m.LSTMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gru = GRUModel().to(device)
    gru.load_state_dict(torch.load(PH2_OUT / "best_gru.pt", weights_only=True))
    gru.eval()

    from collections import defaultdict
    all_hpe = defaultdict(list)  # (model, frac, dur) -> [final_hpe_per_flight]

    SEQ_LEN = 40; FS = 2; FRACS = [0.2, 0.4, 0.6]
    DURS = {"10s": 10, "30s": 30, "60s": 60}
    ENS_W = 0.75

    lstm = LSTMModel().to(device)
    lstm.load_state_dict(torch.load(PH2_OUT / "best_lstm.pt", weights_only=True))
    lstm.eval()

    def make_windows(X_s, N):
        windows = np.zeros((N, SEQ_LEN, 12), dtype=np.float32)
        for t in range(N):
            s = max(0, t - SEQ_LEN + 1)
            w = X_s[s:t+1]
            if len(w) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(w), 12), dtype=np.float32)
                w = np.concatenate([pad, w])
            windows[t] = w
        return windows

    def detrend(win):
        x = win.copy()
        for c in [6, 7, 8, 10]:
            x[:, c] = (x[:, c] - x[:, c].mean()) / (x[:, c].std() + 1e-8)
        return x

    @torch.no_grad()
    def infer(model, windows):
        preds = []
        for i in range(0, len(windows), 256):
            bw = np.stack([detrend(w) for w in windows[i:i+256]])
            preds.append(model(torch.from_numpy(bw).to(device)).cpu().numpy())
        return np.vstack(preds)

    for npz_path in sorted(PH1.glob("flight_*_test.npz")):
        data = np.load(npz_path)
        X_scaled = data["X_scaled"].astype(np.float32)
        y = data["y_delta"].astype(np.float32)
        N = len(X_scaled)
        if N < SEQ_LEN + 20:
            continue

        wins = make_windows(X_scaled, N)
        pg = infer(gru, wins)
        pl = infer(lstm, wins)
        pe = ENS_W * pg + (1-ENS_W) * pl

        for frac in FRACS:
            start = int(N * frac)
            if start >= N - 5:
                continue
            for dur_lbl, dur_s in DURS.items():
                end = min(start + int(dur_s * FS), N)
                if end <= start:
                    continue
                for label, pred in [("GRU", pg), ("LSTM", pl), ("Ensemble", pe)]:
                    cn = ce = 0.0
                    for t in range(start, end):
                        cn += pred[t, 0] - y[t, 0]
                        ce += pred[t, 1] - y[t, 1]
                    all_hpe[(label, frac, dur_lbl)].append(float(np.sqrt(cn**2 + ce**2)))

    return all_hpe


def plot_boxplot(all_hpe):
    """Boxplot: model başına, her süre için tüm uçuş HPE dağılımı"""
    fracs  = [0.2, 0.4, 0.6]
    durs   = ["10s", "30s", "60s"]
    models = ["GRU", "LSTM", "Ensemble"]
    colors = {"GRU": "steelblue", "LSTM": "coral", "Ensemble": "mediumseagreen"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    for ax, dur in zip(axes, durs):
        positions = []
        data_to_plot = []
        tick_labels = []
        pos = 0
        for frac in fracs:
            group_start = pos
            for i, m in enumerate(models):
                vals = all_hpe.get((m, frac, dur), [])
                if vals:
                    bp = ax.boxplot(vals, positions=[pos], widths=0.6,
                                    patch_artist=True,
                                    boxprops=dict(facecolor=colors[m], alpha=0.7),
                                    medianprops=dict(color="black", lw=2),
                                    whiskerprops=dict(color=colors[m]),
                                    capprops=dict(color=colors[m]),
                                    flierprops=dict(marker=".", color=colors[m], alpha=0.5))
                    pos += 1
            # grup boşluğu
            ax.axvline(x=pos - 0.5, color="gray", ls="--", alpha=0.3)
            tick_labels.append(f"%{int(frac*100)}")
            pos += 0.5

        # Legend proxy
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=colors[m], label=m) for m in models]
        ax.legend(handles=legend_els, fontsize=8, loc="upper right")

        # Grup etiketleri
        group_centers = [i * 3.5 + 1 for i in range(len(fracs))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([f"%{int(f*100)}" for f in fracs])
        ax.set(ylabel="Final HPE (m)", title=f"Kesinti Süresi: {dur}")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("GPS Kesintisi Deneyi — HPE Dağılımı (Boxplot)\n"
                 "Her grup: Kesinti başlangıcı (%20/%40/%60) | N=22 test uçuşu", fontsize=11)
    plt.tight_layout()
    plt.savefig(PH4_PLOT / "outage_hpe_boxplot.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Plot: outage_hpe_boxplot.png")


def plot_combined_summary():
    """Phase 2 test + Phase 3 outage kombine özet çubuk grafiği"""
    ph2_metrics = {}
    for fname, label in [("metrics_gru.json", "GRU"), ("metrics_lstm.json", "LSTM")]:
        p = PH2_OUT / fname
        if p.exists():
            with open(p) as f: d = json.load(f)
            ph2_metrics[label] = d

    summary = load_matrix()

    models = ["GRU", "LSTM"]
    scenarios = {
        "Per-Step\n(Phase 2)": lambda m: ph2_metrics.get(m, {}).get("HPE_mean", 0),
        "10s Outage\n(%20 başl.)": lambda m: summary.get(f"{m}|0.2|10s", {}).get("mean", 0),
        "30s Outage\n(%20 başl.)": lambda m: summary.get(f"{m}|0.2|30s", {}).get("mean", 0),
        "60s Outage\n(%20 başl.)": lambda m: summary.get(f"{m}|0.2|60s", {}).get("mean", 0),
    }

    x = np.arange(len(scenarios))
    width = 0.35
    colors_m = {"GRU": "steelblue", "LSTM": "coral"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(models):
        vals = [fn(m) for fn in scenarios.values()]
        bars = ax.bar(x + i*width, vals, width, label=m,
                      color=colors_m[m], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{val:.0f}m", ha="center", fontsize=8)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(list(scenarios.keys()), fontsize=10)
    ax.set(ylabel="HPE (m)", title="GRU vs LSTM — Per-Step ve GPS Kesintisi Karşılaştırması")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PH4_PLOT / "combined_model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Plot: combined_model_comparison.png")


def main():
    print("=" * 60)
    print("PHASE 4 — 02: Boxplot ve Kombine Karsilastirma")
    print("=" * 60)

    print("  Phase 3 verileri yeniden hesaplaniyor (boxplot icin)...")
    all_hpe = load_ph3_raw()

    print("  Boxplot grafigi...")
    plot_boxplot(all_hpe)

    print("  Kombine ozet grafigi (Phase2+Phase3)...")
    plot_combined_summary()

    print("\nAdim 2 tamamlandi.")


if __name__ == "__main__":
    main()
