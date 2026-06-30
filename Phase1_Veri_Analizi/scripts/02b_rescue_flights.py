"""
Phase 1 — Adım 2b: Reddedilen Uçuşları Kurtarma

7 uçuş farklı f-indeksleriyle kurtarılabilir:
  vuelo_7,8,9   : delta_angle f125, baro f118, NED f58 (aynı)
  vuelo_117-120 : delta_angle f129-131, baro f122, NED f61, mag f52-54

Strateji: her dosya için sensör sütunlarını dinamik olarak tespit et,
standart isimlere eşle, mevcut pipeline ile işle.
vuelo_16 (14 satır) ve vuelo_113 (3 satır) kurtarılamaz.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
GROUPED_DIR = BASE_DIR / "Holybro Pixhawk Dataset" / "Holybro Pixhawk" / "processed" / "Grouped flights"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

TIMESTAMP_COL = "timestamp"
SEQUENCE_LENGTH = 40
STEP_SIZE = 4
ACTUAL_HZ = 2.0
MIN_ROWS = 100

# Kurtarılacak uçuşlar ve hangi split'e gidecekleri
RESCUE_MAP = {
    # (delta_angle_base, delta_vel_base, mag_base, baro_col, ned_x, ned_y, ned_z, ref_lat, ref_lon, ref_alt, split)
    "vuelo_7":   ("f125", "f125", "f49", "baro_alt_meter_f118", "x_f58", "y_f58", "z_f58", "ref_lat_f58", "ref_lon_f58", "ref_alt_f58", "train"),
    "vuelo_8":   ("f125", "f125", "f49", "baro_alt_meter_f118", "x_f58", "y_f58", "z_f58", "ref_lat_f58", "ref_lon_f58", "ref_alt_f58", "train"),
    "vuelo_9":   ("f125", "f125", "f49", "baro_alt_meter_f118", "x_f58", "y_f58", "z_f58", "ref_lat_f58", "ref_lon_f58", "ref_alt_f58", "train"),
    "vuelo_117": ("f129", "f129", "f52", "baro_alt_meter_f122", "x_f61", "y_f61", "z_f61", "ref_lat_f61", "ref_lon_f61", "ref_alt_f61", "test"),
    "vuelo_118": ("f129", "f129", "f52", "baro_alt_meter_f122", "x_f61", "y_f61", "z_f61", "ref_lat_f61", "ref_lon_f61", "ref_alt_f61", "test"),
    "vuelo_119": ("f129", "f129", "f52", "baro_alt_meter_f122", "x_f61", "y_f61", "z_f61", "ref_lat_f61", "ref_lon_f61", "ref_alt_f61", "test"),
    "vuelo_120": ("f129", "f129", "f52", "baro_alt_meter_f122", "x_f61", "y_f61", "z_f61", "ref_lat_f61", "ref_lon_f61", "ref_alt_f61", "test"),
}

# Standart girdi sırası (index 0-11)
STANDARD_INPUTS = [
    "delta_angle[0]", "delta_angle[1]", "delta_angle[2]",
    "delta_velocity[0]", "delta_velocity[1]", "delta_velocity[2]",
    "mag_field[0]", "mag_field[1]", "mag_field[2]",
    "indicated_airspeed",
    "baro_alt",
    "diff_pressure",
]


def parse_timestamp_s(ts_series):
    ts = pd.to_datetime(ts_series, errors="coerce")
    return ts.astype(np.int64) / 1e6


def build_column_map(flight_id: str) -> dict:
    """Uçuşa özgü f-indeks eşlemesi döndürür."""
    cfg = RESCUE_MAP[flight_id]
    da_f, dv_f, mag_f, baro_col, nx, ny, nz, rl, rlo, ra, split = cfg

    return {
        "delta_angle[0]": f"delta_angle[0]_{da_f}",
        "delta_angle[1]": f"delta_angle[1]_{da_f}",
        "delta_angle[2]": f"delta_angle[2]_{da_f}",
        "delta_velocity[0]": f"delta_velocity[0]_{dv_f}",
        "delta_velocity[1]": f"delta_velocity[1]_{dv_f}",
        "delta_velocity[2]": f"delta_velocity[2]_{dv_f}",
        "mag_field[0]": f"mag_field[0]_{mag_f}",
        "mag_field[1]": f"mag_field[1]_{mag_f}",
        "mag_field[2]": f"mag_field[2]_{mag_f}",
        "indicated_airspeed": "indicated_airspeed_m_s_f5",
        "baro_alt": baro_col,
        "diff_pressure": "differential_pressure_pa_f15",
        "ned_x": nx, "ned_y": ny, "ned_z": nz,
        "ref_lat": rl, "ref_lon": rlo, "ref_alt": ra,
        "split": split,
    }


def process_rescue_flight(flight_id: str) -> dict | None:
    csv_path = GROUPED_DIR / f"{flight_id}_sync.csv"
    if not csv_path.exists():
        print(f"    DOSYA YOK: {csv_path.name}")
        return None

    col_map = build_column_map(flight_id)

    # Gerekli sütunları topla
    needed = [TIMESTAMP_COL] + list(col_map[k] for k in col_map if not k.startswith("split"))
    needed = list(dict.fromkeys(needed))   # dedupe

    try:
        all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        avail = [c for c in needed if c in all_cols]
        missing = [c for c in needed if c not in all_cols]
        if missing:
            print(f"    [!!] {flight_id}: eksik sutunlar: {missing[:3]}")

        df = pd.read_csv(csv_path, usecols=avail, low_memory=False)
    except Exception as e:
        print(f"    HATA {flight_id}: {e}")
        return None

    if len(df) < MIN_ROWS:
        print(f"    [--] {flight_id}: az satir ({len(df)})")
        return None

    # Timestamp
    df["time_s"] = parse_timestamp_s(df[TIMESTAMP_COL])
    df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

    # Girdi sütunlarını standart isimlerle numpy dizisine al
    X_arr = np.zeros((len(df), 12), dtype=np.float32)
    for i, std_name in enumerate(STANDARD_INPUTS):
        raw_col = col_map.get(std_name)
        if raw_col and raw_col in df.columns:
            vals = pd.to_numeric(df[raw_col], errors="coerce")
            vals = vals.interpolate(method="linear", limit=5).ffill().bfill().fillna(0.0)
            X_arr[:, i] = vals.values.astype(np.float32)

    # NED hedef
    ned_x_col = col_map["ned_x"]
    ned_y_col = col_map["ned_y"]
    ned_z_col = col_map["ned_z"]

    y_abs = np.zeros((len(df), 3), dtype=np.float32)
    for j, col in enumerate([ned_x_col, ned_y_col, ned_z_col]):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.interpolate(method="linear", limit=10).ffill().bfill().fillna(0.0)
            y_abs[:, j] = vals.values.astype(np.float32)

    # Delta NED
    y_delta = np.zeros((len(df), 3), dtype=np.float32)
    y_delta[:, 0] = np.diff(y_abs[:, 0], prepend=y_abs[0, 0])   # ΔNorth
    y_delta[:, 1] = np.diff(y_abs[:, 1], prepend=y_abs[0, 1])   # ΔEast
    y_delta[:, 2] = np.diff(-y_abs[:, 2], prepend=-y_abs[0, 2]) # ΔUp

    # Outlier temizle
    for k in range(3):
        q99 = np.abs(y_delta[:, k]).max()
        thresh = max(q99 * 3 if q99 > 0 else 50, 50.0)
        y_delta[np.abs(y_delta[:, k]) > thresh, k] = 0.0

    # Referans konum
    ref_lat_col = col_map["ref_lat"]
    ref_lon_col = col_map["ref_lon"]
    ref_alt_col = col_map["ref_alt"]
    ref_lat = float(df[ref_lat_col].iloc[0]) if ref_lat_col in df.columns else 0.0
    ref_lon = float(df[ref_lon_col].iloc[0]) if ref_lon_col in df.columns else 0.0
    ref_alt = float(df[ref_alt_col].iloc[0]) if ref_alt_col in df.columns else 0.0

    dur_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0])
    split = col_map["split"]

    print(f"    [OK] {flight_id}: {len(df)} satir, {dur_s:.1f}s → {split}")

    return {
        "flight_id": flight_id,
        "X": X_arr,
        "y_delta": y_delta,
        "y_abs": y_abs,
        "time_s": df["time_s"].values.astype(np.float32),
        "n_rows": len(df),
        "duration_s": dur_s,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "ref_alt": ref_alt,
        "split": split,
    }


def create_sequences(X, y, seq_len=SEQUENCE_LENGTH, step=STEP_SIZE):
    X_seqs, y_seqs = [], []
    for i in range(0, len(X) - seq_len, step):
        X_seqs.append(X[i:i + seq_len])
        y_seqs.append(y[i + seq_len - 1])
    if not X_seqs:
        return np.empty((0, seq_len, X.shape[1]), np.float32), np.empty((0, 3), np.float32)
    return np.array(X_seqs, np.float32), np.array(y_seqs, np.float32)


def verify_rescue_data(rescue_results: list):
    """Kurtarılan verilerin değer aralıklarını mevcut veriyle karşılaştır."""
    print("\n  Veri tutarlılık kontrolü (kurtarılan vs mevcut):")
    X_existing = np.load(OUT_DIR / "X_train.npy")
    existing_mean = X_existing.reshape(-1, 12).mean(axis=0)
    existing_std = X_existing.reshape(-1, 12).std(axis=0)

    for r in rescue_results:
        if r["split"] == "train":
            X_r = r["X"]
            r_mean = X_r.mean(axis=0)
            r_std = X_r.std(axis=0)
            # Oran kontrolü: kurtarılan std, mevcut std ile <5x fark içinde olmalı
            ratio = r_std / (existing_std + 1e-9)
            ok = all(0.1 < v < 10.0 for v in ratio)
            status = "[OK]" if ok else "[!!] DIKKAT: Olcek farki var"
            print(f"    {r['flight_id']}: {status}")
            if not ok:
                for i, (rv, ev) in enumerate(zip(r_std, existing_std)):
                    if not (0.1 < rv/(ev+1e-9) < 10):
                        print(f"      Sutun {i}: rescue_std={rv:.5f}, existing_std={ev:.5f}")


def main():
    print("=" * 60)
    print("ADIM 2b: Reddedilen Ucuslari Kurtarma")
    print("=" * 60)

    print(f"\nKurtarılacak: {list(RESCUE_MAP.keys())}")
    print("Kurtarılamaz: vuelo_16 (14 satir), vuelo_113 (3 satir)\n")

    # Kurtarma işlemi
    rescue_results = []
    for flight_id in RESCUE_MAP:
        print(f"  {flight_id}:")
        result = process_rescue_flight(flight_id)
        if result:
            rescue_results.append(result)

    if not rescue_results:
        print("HATA: Hicbir ucus kurtarilamadi.")
        return

    print(f"\n  Kurtarılan: {len(rescue_results)}/7")

    # Tutarlılık kontrolü
    verify_rescue_data(rescue_results)

    # Mevcut scaler yükle
    with open(OUT_DIR / "scaler.pkl", "rb") as f:
        old_scaler = pickle.load(f)

    # Kurtarılan uçuşları eğitim kümesine katarak scalerı YENİDEN FİT ET
    train_rescued = [r for r in rescue_results if r["split"] == "train"]
    test_rescued = [r for r in rescue_results if r["split"] == "test"]

    print(f"\n  Train'e eklenecek: {[r['flight_id'] for r in train_rescued]}")
    print(f"  Test'e eklenecek : {[r['flight_id'] for r in test_rescued]}")

    # Eğitim veri genişletme
    X_train_orig_flat = np.load(OUT_DIR / "X_train.npy").reshape(-1, 12)
    X_train_rescue_flat = np.vstack([r["X"] for r in train_rescued]) if train_rescued else np.empty((0, 12), np.float32)
    X_train_combined = np.vstack([X_train_orig_flat, X_train_rescue_flat])

    # Scaler yeniden fit
    new_scaler = StandardScaler()
    new_scaler.fit(X_train_combined)
    print(f"\n  Yeni scaler fit: {X_train_combined.shape}")
    print(f"  Mean fark (ilk 3): {(new_scaler.mean_ - old_scaler.mean_)[:3].round(6)}")

    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(new_scaler, f)
    print("  Scaler guncellendi.")

    # Tüm split'leri NPY olarak yeniden oluştur
    print("\n  NPY dizileri guncelleniyor...")
    for split_name, rescued_subset in [("train", train_rescued), ("test", test_rescued), ("val", [])]:
        X_old = np.load(OUT_DIR / f"X_{split_name}.npy")   # (N, seq, 12)
        y_old = np.load(OUT_DIR / f"y_{split_name}.npy")   # (N, 3)

        # Kurtarılan uçuşlardan yeni diziler
        X_new_list, y_new_list = [], []
        for r in rescued_subset:
            X_sc = new_scaler.transform(r["X"])
            X_seq, y_seq = create_sequences(X_sc, r["y_delta"])
            if len(X_seq) > 0:
                X_new_list.append(X_seq)
                y_new_list.append(y_seq)

        # Mevcut dizileri de yeni scaler ile yeniden ölçekle
        # (split = train: tüm train datası yeniden ölçeklenmeli)
        if split_name == "train":
            # Mevcut sekansları ham veriden yeniden oluşturmak yerine,
            # orjinal ve yeni scaler arasındaki dönüşümü uygula
            # X_old shape: (N, 40, 12) — X_scaled = (X_raw - old_mean) / old_std
            # X_raw = X_old * old_std + old_mean
            # X_new_scaled = (X_raw - new_mean) / new_std
            X_old_flat = X_old.reshape(-1, 12)
            X_raw_approx = X_old_flat * old_scaler.scale_ + old_scaler.mean_
            X_rescaled = new_scaler.transform(X_raw_approx).reshape(X_old.shape).astype(np.float32)
        else:
            # Val/test: aynı mantık
            X_old_flat = X_old.reshape(-1, 12)
            X_raw_approx = X_old_flat * old_scaler.scale_ + old_scaler.mean_
            X_rescaled = new_scaler.transform(X_raw_approx).reshape(X_old.shape).astype(np.float32)

        # Birleştir
        if X_new_list:
            X_extra = np.vstack(X_new_list)
            y_extra = np.vstack(y_new_list)
            X_final = np.vstack([X_rescaled, X_extra])
            y_final = np.vstack([y_old, y_extra])
        else:
            X_final = X_rescaled
            y_final = y_old

        np.save(OUT_DIR / f"X_{split_name}.npy", X_final)
        np.save(OUT_DIR / f"y_{split_name}.npy", y_final)
        print(f"  {split_name}: {X_old.shape} -> {X_final.shape}")

    # Ham NPZ dosyaları kaydet
    for r in rescue_results:
        np.savez_compressed(
            OUT_DIR / f"flight_{r['flight_id']}_{r['split']}.npz",
            X_raw=r["X"],
            X_scaled=new_scaler.transform(r["X"]),
            y_delta=r["y_delta"],
            y_abs=r["y_abs"],
            time_s=r["time_s"],
        )

    # flight_splits.csv güncelle
    splits_df = pd.read_csv(OUT_DIR / "flight_splits.csv")
    for r in rescue_results:
        new_row = pd.DataFrame([{
            "file": f"{r['flight_id']}_sync.csv",
            "flight_id": r["flight_id"],
            "path": str(GROUPED_DIR / f"{r['flight_id']}_sync.csv"),
            "row_count": r["n_rows"],
            "duration_s": r["duration_s"],
            "sample_hz": 2.0,
            "split": r["split"],
        }])
        splits_df = pd.concat([splits_df, new_row], ignore_index=True)
    splits_df.to_csv(OUT_DIR / "flight_splits.csv", index=False)

    # flight_meta.csv güncelle
    meta_df = pd.read_csv(OUT_DIR / "flight_meta.csv")
    for r in rescue_results:
        new_meta = pd.DataFrame([{
            "flight_id": r["flight_id"],
            "split": r["split"],
            "n_rows": r["n_rows"],
            "duration_s": r["duration_s"],
            "ref_lat": r["ref_lat"],
            "ref_lon": r["ref_lon"],
            "ref_alt": r["ref_alt"],
        }])
        meta_df = pd.concat([meta_df, new_meta], ignore_index=True)
    meta_df.to_csv(OUT_DIR / "flight_meta.csv", index=False)

    # Özet
    print(f"\n{'='*60}")
    print(f"KURTARMA TAMAMLANDI")
    print(f"  Kurtarılan ucus : {len(rescue_results)}/7")
    print(f"  Toplam gecerli  : {len(splits_df)} (onceki: {len(splits_df)-len(rescue_results)})")
    for s in ["train","val","test"]:
        X = np.load(OUT_DIR / f"X_{s}.npy")
        print(f"  {s}: {X.shape}")
    print("\nAdim 2b tamamlandi.")

    return {"rescued": [r["flight_id"] for r in rescue_results]}


if __name__ == "__main__":
    main()
