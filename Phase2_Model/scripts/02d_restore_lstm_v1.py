"""
LSTM v1 yeniden eğitim — orijinal hiperparametreler.
02c ile best_lstm.pt bozuldu, bu script onu restore eder.
"""
import numpy as np, torch, torch.nn as nn, time, json, importlib.util
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = Path(__file__).resolve().parents[2]
PH1      = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2      = Path(__file__).resolve().parents[1] / "outputs"
PLOT_DIR = PH2 / "plots"

_spec = importlib.util.spec_from_file_location("models", Path(__file__).parent / "models.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
detrend_windows = _m.detrend_windows
LSTMModel = _m.LSTMModel   # orijinal v1 mimarisi (hidden=128)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)
    print("=" * 60)
    print(f"LSTM v1 RESTORE  [{device}]  hidden=128, ReduceLROnPlateau")
    print("=" * 60)

    X_tr = torch.from_numpy(detrend_windows(np.load(PH1 / "X_train.npy")))
    y_tr = torch.from_numpy(np.load(PH1 / "y_train.npy"))
    X_va = torch.from_numpy(detrend_windows(np.load(PH1 / "X_val.npy")))
    y_va = torch.from_numpy(np.load(PH1 / "y_val.npy"))

    train_ld = DataLoader(TensorDataset(X_tr, y_tr), 256, shuffle=True,  pin_memory=True)
    val_ld   = DataLoader(TensorDataset(X_va, y_va), 512, shuffle=False, pin_memory=True)

    model     = LSTMModel().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametre: {n_params:,}")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    best_val = 1e9; best_ep = 0; no_imp = 0
    tr_ls, va_ls = [], []
    t0 = time.time()
    SAVE = PH2 / "best_lstm.pt"

    for ep in range(1, 151):
        model.train()
        tl = []
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for Xb, yb in val_ld:
                vl.append(criterion(model(Xb.to(device)), yb.to(device)).item())

        tr = float(np.mean(tl)); va = float(np.mean(vl))
        tr_ls.append(tr); va_ls.append(va)
        scheduler.step(va)

        if va < best_val:
            best_val = va; best_ep = ep; no_imp = 0
            torch.save(model.state_dict(), SAVE)
        else:
            no_imp += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  {ep:4d} | tr={tr:.5f} | va={va:.5f} | best={best_val:.5f}@{best_ep} | {time.time()-t0:.0f}s")

        if no_imp >= 15:
            print(f"  Early stop @ ep {ep}")
            break

    print(f"\n  LSTM_v1 restore: val={best_val:.5f} @ ep {best_ep}")

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
    print(f"  Test HPE: mean={np.mean(hpe):.3f}  median={np.median(hpe):.3f}  p95={np.percentile(hpe,95):.3f}")

    meta = {
        "model": "LSTM", "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
        "best_val_loss": float(best_val), "best_epoch": best_ep,
        "n_params": n_params, "device": str(device),
    }
    with open(PH2 / "lstm_training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr_ls, label="Train"); ax.plot(va_ls, label="Val", color="coral")
    ax.axvline(best_ep-1, color="r", ls="--", label=f"Best ep={best_ep}")
    ax.set(xlabel="Epoch", ylabel="Huber Loss", title="LSTM v1 restore (hidden=128)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lstm_training_loss.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  best_lstm.pt restore edildi.")
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
