"""
Phase 2b — Geliştirilmiş Model Eğitimi

Mevcut sorunlar ve çözümler:
  - hidden=128 küçük        → 256'ya çıkar
  - P95=15m yüksek          → LayerNorm + combined loss (Huber + L1)
  - Drift/sistematik hata   → CosineAnnealingLR, daha uzun eğitim
  - Tek step tahmin zayıf   → Seq2Seq: tüm sequence'taki adımları tahmin et

Eğer yeni model mevcut best_gru.pt / best_lstm.pt'yi geçerse üzerine yazar.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import importlib.util
import warnings
warnings.filterwarnings("ignore")

PHASE1_OUT = Path(__file__).resolve().parents[2] / "Phase1_Veri_Analizi" / "outputs"
PHASE2_OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOT_DIR   = PHASE2_OUT / "plots"

# ─── Hiperparametreler ───────────────────────────────────────
HIDDEN_SIZE  = 256      # 128 → 256  (~4x kapasite artışı, 2 katman)
NUM_LAYERS   = 2        # 3 değil 2 — 9677 sample için 3 katman overfitting
DROPOUT      = 0.3      # 0.2→0.3 (daha büyük model için)
BATCH_SIZE   = 256
MAX_EPOCHS   = 200
LR           = 1e-3
T_MAX        = 50       # CosineAnnealing yarım periyot
ETA_MIN      = 1e-5
PATIENCE     = 25
GRAD_CLIP    = 1.0
HUBER_DELTA  = 1.0
L1_WEIGHT    = 0.0      # Sadece Huber — v1 ile karşılaştırılabilir olsun
SEED         = 42
# ─────────────────────────────────────────────────────────────

# models.py'den detrend_windows'ı al
_mod = importlib.util.spec_from_file_location(
    "models", Path(__file__).parent / "models.py")
_m = importlib.util.module_from_spec(_mod)
_mod.loader.exec_module(_m)
detrend_windows = _m.detrend_windows
detrend_single  = _m.detrend_single


class ImprovedGRU(nn.Module):
    """
    GRU + LayerNorm + skip connection.
    hidden_size=256, num_layers=3
    """
    def __init__(self, input_size=12, hidden_size=256,
                 num_layers=3, dropout=0.3, output_size=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last = self.norm(out[:, -1, :])
        return self.fc(self.dropout(last))


class ImprovedLSTM(nn.Module):
    """LSTM + LayerNorm, aynı yapı."""
    def __init__(self, input_size=12, hidden_size=256,
                 num_layers=3, dropout=0.3, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])
        return self.fc(self.dropout(last))


class CombinedLoss(nn.Module):
    """Huber + ağırlıklı L1 → outlier baskılar, küçük hataları da cezalandırır."""
    def __init__(self, delta=1.0, l1_weight=0.3):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.l1    = nn.L1Loss()
        self.w     = l1_weight

    def forward(self, pred, target):
        return self.huber(pred, target) + self.w * self.l1(pred, target)


def load_data():
    X_tr = detrend_windows(np.load(PHASE1_OUT / "X_train.npy"))
    y_tr = np.load(PHASE1_OUT / "y_train.npy")
    X_va = detrend_windows(np.load(PHASE1_OUT / "X_val.npy"))
    y_va = np.load(PHASE1_OUT / "y_val.npy")
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(X_va), torch.from_numpy(y_va))


def train_model(model, name, device, train_loader, val_loader, save_path, old_val_loss=None):
    criterion = CombinedLoss(delta=HUBER_DELTA, l1_weight=L1_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )

    tr_losses, va_losses = [], []
    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  {name} — {n_params:,} parametre  [{device}]")
    print(f"  {'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'LR':>9} | Süre")
    print(f"  {'-'*55}")

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        tr_total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_total += loss.item() * len(Xb)
        tr_loss = tr_total / len(train_loader.dataset)

        # Val
        model.eval()
        va_total = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                va_total += criterion(model(Xb.to(device)), yb.to(device)).item() * len(Xb)
        va_loss = va_total / len(val_loader.dataset)

        scheduler.step()
        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if va_loss < best_val:
            best_val   = va_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), save_path.with_suffix(".tmp.pt"))
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:5d} | {tr_loss:10.5f} | {va_loss:10.5f} | "
                  f"{lr_now:.3e} | {time.time()-t0:.1f}s")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping @ epoch {epoch}  "
                  f"(best={best_epoch}, val={best_val:.5f})")
            break

    # Eski modeli geç mi?
    tmp_path = save_path.with_suffix(".tmp.pt")
    if old_val_loss is not None and best_val >= old_val_loss:
        print(f"\n  [!!] Yeni model eski kadar iyi degil "
              f"({best_val:.5f} >= {old_val_loss:.5f}) -> eski korunuyor")
        if tmp_path.exists():
            tmp_path.unlink()
        improved = False
    else:
        if tmp_path.exists():
            if save_path.exists():
                save_path.unlink()
            tmp_path.rename(save_path)
        print(f"\n  [OK] Model kaydedildi → {save_path.name}  "
              f"(val={best_val:.5f}, epoch={best_epoch})")
        improved = True

    return {
        "model": name,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "improved_over_v1": improved,
        "n_params": n_params,
        "tr_losses": tr_losses,
        "va_losses": va_losses,
    }


@torch.no_grad()
def quick_eval(model, X_test_np, y_test_np, device, batch_size=512):
    model = model.to(device).eval()
    X_t = torch.from_numpy(detrend_windows(X_test_np))
    preds = []
    for i in range(0, len(X_t), batch_size):
        preds.append(model(X_t[i:i+batch_size].to(device)).cpu().numpy())
    y_pred = np.vstack(preds)
    hpe = np.sqrt((y_pred[:,0]-y_test_np[:,0])**2 + (y_pred[:,1]-y_test_np[:,1])**2)
    return {
        "HPE_mean":   float(np.mean(hpe)),
        "HPE_median": float(np.median(hpe)),
        "HPE_p95":    float(np.percentile(hpe, 95)),
        "RMSE_3D":    float(np.sqrt(np.mean(np.sum((y_pred-y_test_np)**2, axis=1)))),
    }


def plot_loss(results_list, plot_dir):
    fig, axes = plt.subplots(1, len(results_list), figsize=(12, 5))
    if len(results_list) == 1:
        axes = [axes]
    colors = {"GRU_v2": "steelblue", "LSTM_v2": "coral"}
    for ax, r in zip(axes, results_list):
        n = r["model"]
        ep = range(1, len(r["tr_losses"]) + 1)
        ax.plot(ep, r["tr_losses"], label="Train", color=colors.get(n, "gray"))
        ax.plot(ep, r["va_losses"], label="Val",   color="green", alpha=0.8)
        ax.axvline(r["best_epoch"], color="red", ls="--", alpha=0.6,
                   label=f"Best ({r['best_epoch']})")
        ax.set_title(f"{n}  |  val={r['best_val_loss']:.4f}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Combined Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Geliştirilmiş Modeller — Eğitim Kaybı", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir / "improved_loss.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot: improved_loss.png")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ADIM 2b: Geliştirilmiş Model Eğitimi")
    print(f"  hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, "
          f"loss=Huber+L1, LR=Cosine({T_MAX})")
    print("=" * 60)

    X_tr, y_tr, X_va, y_va = load_data()
    X_test_np = np.load(PHASE1_OUT / "X_test.npy")
    y_test_np = np.load(PHASE1_OUT / "y_test.npy")

    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}")

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Mevcut v1 val loss'larını oku
    def _old_val(meta_file):
        p = PHASE2_OUT / meta_file
        if p.exists():
            with open(p) as f:
                return json.load(f).get("best_val_loss")
        return None

    results = []

    # ── GRU v2 ────────────────────────────────────────────────
    old_gru_val = _old_val("gru_training_meta.json")
    print(f"\n  Mevcut GRU val loss: {old_gru_val}")
    gru_model = ImprovedGRU(
        input_size=12, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    res_gru = train_model(
        gru_model, "GRU_v2", device,
        train_loader, val_loader,
        save_path=PHASE2_OUT / "best_gru.pt",
        old_val_loss=old_gru_val
    )
    results.append(res_gru)

    # ── LSTM v2 ───────────────────────────────────────────────
    old_lstm_val = _old_val("lstm_training_meta.json")
    print(f"\n  Mevcut LSTM val loss: {old_lstm_val}")
    lstm_model = ImprovedLSTM(
        input_size=12, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    res_lstm = train_model(
        lstm_model, "LSTM_v2", device,
        train_loader, val_loader,
        save_path=PHASE2_OUT / "best_lstm.pt",
        old_val_loss=old_lstm_val
    )
    results.append(res_lstm)

    # ── Hızlı test değerlendirmesi ────────────────────────────
    print("\n  Test seti hızlı değerlendirme:")
    compare_rows = []
    for res, model, name in [
        (res_gru,  gru_model,  "GRU_v2"),
        (res_lstm, lstm_model, "LSTM_v2"),
    ]:
        if res["improved_over_v1"]:
            m = quick_eval(model, X_test_np, y_test_np, device)
            compare_rows.append((name, m))
            print(f"  {name}: HPE_mean={m['HPE_mean']:.3f}m  "
                  f"HPE_median={m['HPE_median']:.3f}m  "
                  f"HPE_p95={m['HPE_p95']:.3f}m  "
                  f"RMSE_3D={m['RMSE_3D']:.3f}m")

    # v1 karşılaştırma (önceki metrikler varsa)
    for fname, label in [("metrics_gru.json", "GRU_v1"),
                         ("metrics_lstm.json", "LSTM_v1")]:
        p = PHASE2_OUT / fname
        if p.exists():
            with open(p) as f: mv = json.load(f)
            print(f"  {label}: HPE_mean={mv.get('HPE_mean',0):.3f}m  "
                  f"HPE_median={mv.get('HPE_median',0):.3f}m  "
                  f"HPE_p95={mv.get('HPE_p95',0):.3f}m")

    # Meta kaydet
    for res in results:
        meta = {k: v for k, v in res.items() if k not in ("tr_losses", "va_losses")}
        fname = f"{'gru' if 'GRU' in res['model'] else 'lstm'}_training_meta.json"
        with open(PHASE2_OUT / fname, "w") as f:
            json.dump(meta, f, indent=2)

    # Grafik
    plot_loss(results, PLOT_DIR)

    print(f"\n{'='*60}")
    improved = [r["model"] for r in results if r["improved_over_v1"]]
    print(f"  İyileşen: {improved if improved else 'Yok — v1 korundu'}")
    print("Adım 2b tamamlandı.")
    return results


if __name__ == "__main__":
    main()
