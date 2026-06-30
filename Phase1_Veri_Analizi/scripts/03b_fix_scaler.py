"""
Adım 3b: Scaler Düzeltme
02b_rescue_flights.py'deki bug: X_train.npy (scaled) üzerinden scaler fit edildi.
Tüm ham NPZ dosyalarından raw data yüklenerek scaler yeniden fit edilir
ve numpy dizileri doğru şekilde yeniden üretilir.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SEQUENCE_LENGTH = 40
STEP_SIZE = 4


def create_sequences(X, y, seq_len=SEQUENCE_LENGTH, step=STEP_SIZE):
    X_seqs, y_seqs = [], []
    for i in range(0, len(X) - seq_len, step):
        X_seqs.append(X[i:i + seq_len])
        y_seqs.append(y[i + seq_len - 1])
    if not X_seqs:
        return np.empty((0, seq_len, X.shape[1]), np.float32), np.empty((0, 3), np.float32)
    return np.array(X_seqs, np.float32), np.array(y_seqs, np.float32)


def main():
    print("=" * 60)
    print("ADIM 3b: Scaler Düzeltme ve NPY Yeniden Üretim")
    print("=" * 60)

    # Tüm split NPZ dosyalarını bul
    train_files = sorted(OUT_DIR.glob("flight_*_train.npz"))
    val_files   = sorted(OUT_DIR.glob("flight_*_val.npz"))
    test_files  = sorted(OUT_DIR.glob("flight_*_test.npz"))

    print(f"\nNPZ dosyaları: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Ham eğitim verisi yükle → scaler fit
    print("\n  Ham train verisi yükleniyor...")
    X_train_raw_list = []
    for f in train_files:
        data = np.load(f)
        X_train_raw_list.append(data["X_raw"].astype(np.float32))
    X_train_raw_all = np.vstack(X_train_raw_list)
    print(f"  Ham train shape: {X_train_raw_all.shape}")
    print(f"  delta_angle[0] ham std: {X_train_raw_all[:,0].std():.6f}")
    print(f"  baro_alt ham std: {X_train_raw_all[:,10].std():.4f}")
    print(f"  diff_press ham std: {X_train_raw_all[:,11].std():.4f}")

    # Düzgün scaler fit
    scaler = StandardScaler()
    scaler.fit(X_train_raw_all)
    print(f"\n  Yeni scaler fit tamamlandı")
    print(f"  scale_[:3]: {scaler.scale_[:3].round(6)}")
    print(f"  scale_[10]: {scaler.scale_[10]:.4f} (baro)")
    print(f"  scale_[11]: {scaler.scale_[11]:.4f} (diff_press)")

    # Scaler kaydet
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  scaler.pkl kaydedildi")

    # Her split için NPY yeniden üret
    for split_name, npz_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"\n  {split_name.upper()} ({len(npz_files)} uçuş) dizileri oluşturuluyor...")
        X_seqs_all, y_seqs_all = [], []
        for f in npz_files:
            data = np.load(f)
            X_raw = data["X_raw"].astype(np.float32)
            y_delta = data["y_delta"].astype(np.float32)

            X_scaled = scaler.transform(X_raw).astype(np.float32)
            X_seq, y_seq = create_sequences(X_scaled, y_delta)
            if len(X_seq) > 0:
                X_seqs_all.append(X_seq)
                y_seqs_all.append(y_seq)

            # NPZ'deki X_scaled'ı da güncelle
            np.savez_compressed(f,
                X_raw=X_raw,
                X_scaled=X_scaled,
                y_delta=y_delta,
                y_abs=data["y_abs"],
                time_s=data["time_s"],
            )

        if X_seqs_all:
            X_final = np.vstack(X_seqs_all).astype(np.float32)
            y_final = np.vstack(y_seqs_all).astype(np.float32)
        else:
            X_final = np.empty((0, SEQUENCE_LENGTH, X_train_raw_all.shape[1]), np.float32)
            y_final = np.empty((0, 3), np.float32)

        np.save(OUT_DIR / f"X_{split_name}.npy", X_final)
        np.save(OUT_DIR / f"y_{split_name}.npy", y_final)
        print(f"  {split_name}: X={X_final.shape}, y={y_final.shape}, NaN={np.isnan(X_final).sum()}")

    # Doğrulama
    print("\n  Son doğrulama:")
    X_tr = np.load(OUT_DIR / "X_train.npy").reshape(-1, 12)
    print(f"  X_train scaled mean[:3]: {X_tr.mean(0)[:3].round(4)}")
    print(f"  X_train scaled std[:3]:  {X_tr.std(0)[:3].round(4)}")
    print(f"  X_train scaled mean[10,11]: {X_tr.mean(0)[10:12].round(4)}")
    print(f"  X_train scaled std[10,11]:  {X_tr.std(0)[10:12].round(4)}")
    print("  (mean~0, std~1 olmalı)")

    print("\nAdım 3b tamamlandı.")


if __name__ == "__main__":
    main()
