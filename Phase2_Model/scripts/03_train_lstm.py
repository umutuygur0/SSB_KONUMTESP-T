"""
Phase 2 — Adım 3: LSTM Model Eğitimi
02_train_gru.py ile aynı mimari/hiperparametreler, GRU → LSTM.
BiLSTM kullanılmıyor — gerçek zamanlı inference için uygun değil.
"""

import sys
import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from models import LSTMModel, detrend_windows

BASE_DIR   = Path(__file__).resolve().parents[2]
PHASE1_OUT = BASE_DIR / "Phase1_Veri_Analizi" / "outputs"
PHASE2_OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS_DIR  = PHASE2_OUT / "plots"
PHASE2_OUT.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
MAX_EPOCHS = 150
LR         = 1e-3
PATIENCE_LR = 5
PATIENCE_ES = 15
HUBER_DELTA = 1.0
GRAD_CLIP   = 1.0


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def plot_training(history: dict, best_epoch: int, model_name: str = "LSTM"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="darkorange")
    ax1.axvline(best_epoch, color="red", ls="--", alpha=0.8, label=f"Best @ {best_epoch}")
    ax1.set_ylabel("Huber Loss")
    ax1.set_title(f"{model_name} — Train / Validation Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.semilogy(epochs, history["lr"], color="seagreen")
    ax2.set_ylabel("Learning Rate (log)"); ax2.set_xlabel("Epoch")
    ax2.set_title(f"{model_name} — Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"{model_name.lower()}_training_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def train():
    print("=" * 60)
    print("ADIM 3: LSTM Model Eğitimi")
    print(f"  Device : {DEVICE}")
    print(f"  Batch  : {BATCH_SIZE} | LR: {LR} | Max Epoch: {MAX_EPOCHS}")
    print("=" * 60)

    X_train = detrend_windows(np.load(PHASE1_OUT / "X_train.npy"))
    y_train = np.load(PHASE1_OUT / "y_train.npy")
    X_val   = detrend_windows(np.load(PHASE1_OUT / "X_val.npy"))
    y_val   = np.load(PHASE1_OUT / "y_val.npy")
    print(f"  Train : X={X_train.shape}  y={y_train.shape}")
    print(f"  Val   : X={X_val.shape}    y={y_val.shape}")
    print(f"  Detrend: mag(6,7,8) + baro(10) within-window DC removal\n")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    model    = LSTMModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LSTM parametreleri: {n_params:,}\n")

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE_LR, factor=0.5
    )

    best_val   = float("inf")
    best_ep    = 0
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [], "lr": []}
    t0         = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        tr_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        vl_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl_losses.append(criterion(model(Xb.to(DEVICE)), yb.to(DEVICE)).item())

        tr = float(np.mean(tr_losses))
        vl = float(np.mean(vl_losses))
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["lr"].append(lr)

        scheduler.step(vl)

        if vl < best_val:
            best_val = vl; best_ep = epoch; no_improve = 0
            torch.save(model.state_dict(), PHASE2_OUT / "best_lstm.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"train={tr:.5f} | val={vl:.5f} | "
                  f"best={best_val:.5f}@{best_ep} | "
                  f"lr={lr:.2e} | {elapsed:.0f}s")

        if no_improve >= PATIENCE_ES:
            print(f"\n  Early stopping: {PATIENCE_ES} epoch iyileşme yok.")
            break

    total_s = time.time() - t0
    print(f"\n  Eğitim tamamlandı: {total_s:.1f}s  (best val={best_val:.5f} @ epoch {best_ep})")

    meta = {
        "model": "LSTM",
        "best_val_loss": best_val,
        "best_epoch": best_ep,
        "total_epochs": len(history["train_loss"]),
        "training_time_s": round(total_s, 1),
        "n_params": n_params,
        "device": str(DEVICE),
        "hyperparams": {
            "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
            "batch_size": BATCH_SIZE, "lr_init": LR,
            "huber_delta": HUBER_DELTA, "patience_es": PATIENCE_ES,
        },
    }
    with open(PHASE2_OUT / "lstm_training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    plot_training(history, best_ep, model_name="LSTM")
    print("Adım 3 tamamlandı.\n")
    return meta


if __name__ == "__main__":
    train()
