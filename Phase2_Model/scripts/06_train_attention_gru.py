"""
Phase 2 — Adım 6: Attention-GRU Eğitimi
GRU + self-attention pooling. Son hidden yerine tüm timestep ağırlıklı ortalaması.
"""
import sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from models import AttentionGRUModel, detrend_windows

BASE_DIR   = Path(__file__).resolve().parents[2]
PHASE1_OUT = BASE_DIR / "Phase1_Veri_Analizi" / "outputs"
PHASE2_OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS_DIR  = PHASE2_OUT / "plots"
PHASE2_OUT.mkdir(exist_ok=True); PLOTS_DIR.mkdir(exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 256
MAX_EPOCHS  = 150
LR          = 1e-3
PATIENCE_LR = 5
PATIENCE_ES = 15
HUBER_DELTA = 1.0
GRAD_CLIP   = 1.0


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def train():
    print("=" * 60)
    print("ADIM 6: Attention-GRU Eğitimi")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    X_tr = detrend_windows(np.load(PHASE1_OUT / "X_train.npy"))
    y_tr = np.load(PHASE1_OUT / "y_train.npy")
    X_vl = detrend_windows(np.load(PHASE1_OUT / "X_val.npy"))
    y_vl = np.load(PHASE1_OUT / "y_val.npy")
    print(f"  Train: {X_tr.shape}  Val: {X_vl.shape}\n")

    tr_loader = make_loader(X_tr, y_tr, shuffle=True)
    vl_loader = make_loader(X_vl, y_vl, shuffle=False)

    model   = AttentionGRUModel().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  AttentionGRU parametreleri: {n_param:,}\n")

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE_LR, factor=0.5)

    best_val, best_ep, no_improve = float("inf"), 0, 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses = []
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for Xb, yb in vl_loader:
                vl_losses.append(criterion(model(Xb.to(DEVICE)), yb.to(DEVICE)).item())

        tr = float(np.mean(tr_losses))
        vl = float(np.mean(vl_losses))
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["lr"].append(lr)
        scheduler.step(vl)

        if vl < best_val:
            best_val, best_ep, no_improve = vl, epoch, 0
            torch.save(model.state_dict(), PHASE2_OUT / "best_attn_gru.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train={tr:.5f} | val={vl:.5f} | "
                  f"best={best_val:.5f}@{best_ep} | lr={lr:.2e} | {time.time()-t0:.0f}s")

        if no_improve >= PATIENCE_ES:
            print(f"\n  Early stopping: {PATIENCE_ES} epoch iyileşme yok.")
            break

    total_s = time.time() - t0
    print(f"\n  Tamamlandı: {total_s:.1f}s  best_val={best_val:.5f} @ epoch {best_ep}")

    # Loss grafiği
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs, history["train_loss"], label="Train", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="darkorange")
    ax1.axvline(best_ep, color="red", ls="--", alpha=0.8, label=f"Best@{best_ep}")
    ax1.set_ylabel("Huber Loss"); ax1.set_title("AttentionGRU — Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.semilogy(epochs, history["lr"], color="seagreen")
    ax2.set_ylabel("LR (log)"); ax2.set_xlabel("Epoch")
    ax2.set_title("Learning Rate Schedule"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attn_gru_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    meta = {
        "model": "AttentionGRU",
        "best_val_loss": best_val,
        "best_epoch": best_ep,
        "total_epochs": len(history["train_loss"]),
        "training_time_s": round(total_s, 1),
        "n_params": n_param,
        "device": str(DEVICE),
        "hyperparams": {
            "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
            "batch_size": BATCH_SIZE, "lr_init": LR,
            "huber_delta": HUBER_DELTA, "patience_es": PATIENCE_ES,
            "attention": "additive (Linear 128→1, softmax over T=40)",
        },
    }
    with open(PHASE2_OUT / "attn_gru_training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    train()
