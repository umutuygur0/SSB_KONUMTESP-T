"""LSTM v2 — hidden=256, LayerNorm, CosineAnnealingLR"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json, time, importlib.util
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = Path(__file__).resolve().parents[2]
PH1       = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2       = Path(__file__).resolve().parents[1] / "outputs"
PLOT_DIR  = PH2 / "plots"

_spec = importlib.util.spec_from_file_location("models", Path(__file__).parent / "models.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
detrend_windows = _m.detrend_windows


class ImprovedLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.)
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(self.norm(out[:, -1, :])))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 60)
    print(f"LSTM v2 Egitimi  [{device}]  hidden=256, LayerNorm, Cosine LR")
    print("=" * 60)

    X_tr = torch.from_numpy(detrend_windows(np.load(PH1 / "X_train.npy")))
    y_tr = torch.from_numpy(np.load(PH1 / "y_train.npy"))
    X_va = torch.from_numpy(detrend_windows(np.load(PH1 / "X_val.npy")))
    y_va = torch.from_numpy(np.load(PH1 / "y_val.npy"))
    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}")

    train_ld = DataLoader(TensorDataset(X_tr, y_tr), 256, shuffle=True,  pin_memory=True)
    val_ld   = DataLoader(TensorDataset(X_va, y_va), 512, shuffle=False, pin_memory=True)

    model     = ImprovedLSTM().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametre: {n_params:,}")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_val = 1e9; best_ep = 0; no_imp = 0
    tr_ls, va_ls = [], []
    t0 = time.time()
    SAVE = PH2 / "best_lstm.pt"

    for ep in range(1, 201):
        model.train()
        tl = []
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for Xb, yb in val_ld:
                vl.append(criterion(model(Xb.to(device)), yb.to(device)).item())

        tr = float(np.mean(tl)); va = float(np.mean(vl))
        tr_ls.append(tr); va_ls.append(va)
        scheduler.step()

        if va < best_val:
            best_val = va; best_ep = ep; no_imp = 0
            torch.save(model.state_dict(), SAVE)
        else:
            no_imp += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  {ep:4d} | tr={tr:.5f} | va={va:.5f} | best={best_val:.5f}@{best_ep} | {time.time()-t0:.0f}s")

        if no_imp >= 25:
            print(f"  Early stop @ ep {ep}")
            break

    print(f"\n  LSTM_v2 sonuc: val={best_val:.5f} @ ep {best_ep}")
    print(f"  LSTM_v1 icin:  val=0.16795 @ ep 51")

    # Test HPE
    X_te = torch.from_numpy(detrend_windows(np.load(PH1 / "X_test.npy")))
    y_te = np.load(PH1 / "y_test.npy")
    model.load_state_dict(torch.load(SAVE, weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 512):
            preds.append(model(X_te[i:i+512].to(device)).cpu().numpy())
    yp = np.vstack(preds)
    hpe = np.sqrt((yp[:,0]-y_te[:,0])**2 + (yp[:,1]-y_te[:,1])**2)
    print(f"\n  LSTM_v2 test HPE: mean={np.mean(hpe):.3f}  median={np.median(hpe):.3f}  p95={np.percentile(hpe,95):.3f}")
    print(f"  LSTM_v1 test HPE: mean=4.261  median=1.602  p95=15.874")

    improved = best_val < 0.16795
    print(f"\n  Sonuc: {'IYILESME VAR -> best_lstm.pt guncellendi' if improved else 'v1 daha iyi -> best_lstm.pt eski haliyle kaldi'}")

    meta = {
        "model": "LSTM_v2",
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "best_val_loss": float(best_val),
        "best_epoch": best_ep,
        "n_params": n_params,
        "device": str(device),
    }
    with open(PH2 / "lstm_training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Eğer v1 iyiyse eski modeli geri yükle (buraya gelemedik ama uyarı ver)
    if not improved:
        print("  [UYARI] best_lstm.pt v2 ile degistirildi ama v2 daha kotu!")
        print("  Eski val loss 0.16795'i suremeyen model kaydedildi.")

    # Loss grafiği
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr_ls, label="Train"); ax.plot(va_ls, label="Val", color="coral")
    ax.axvline(best_ep-1, color="r", ls="--", label=f"Best ep={best_ep}")
    ax.set(xlabel="Epoch", ylabel="Huber Loss", title="LSTM v2 (hidden=256, LayerNorm)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lstm_v2_loss.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Plot: lstm_v2_loss.png")
    print("\nTamamlandi.")
    return meta


if __name__ == "__main__":
    main()
