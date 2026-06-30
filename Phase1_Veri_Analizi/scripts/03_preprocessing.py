"""
Phase 1 — Adım 3: Ön İşleme Pipeline (Düzeltilmiş)

Veri gerçeği: ~2 Hz senkronize zaman serisi, ~500 satır/uçuş (~250 saniye).
Timestamp: pd.to_datetime().astype(int64) = µs boot zamanı.

Yapılan işlemler:
1. Geçerli uçuşları yükle (INPUT + TARGET sütunları)
2. Timestamp'i saniyeye çevir
3. NaN interpolasyonu / forward-fill
4. NED delta pozisyon hesapla (ΔNorth, ΔEast, ΔUp)
5. StandardScaler — sadece train setinden fit
6. Sliding window dizileri oluştur
7. Numpy array olarak kaydet
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
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TIMESTAMP_COL = "timestamp"
ACTUAL_HZ = 2.0          # ~2 Hz, veri zaten senkronize

# Sequence parametreleri @ ~2 Hz
SEQUENCE_LENGTH = 40      # 20 saniye pencere
STEP_SIZE = 4             # 2 saniye adım

INPUT_COLS = [
    "delta_angle[0]_f124", "delta_angle[1]_f124", "delta_angle[2]_f124",
    "delta_velocity[0]_f124", "delta_velocity[1]_f124", "delta_velocity[2]_f124",
    "mag_field[0]_f49", "mag_field[1]_f49", "mag_field[2]_f49",
    "indicated_airspeed_m_s_f5",
    "baro_alt_meter_f117",
    "differential_pressure_pa_f15",
]

TARGET_COLS = ["x_f58", "y_f58", "z_f58"]  # NED local position (metre)
REF_COLS = ["ref_lat_f58", "ref_lon_f58", "ref_alt_f58"]


def parse_timestamp_s(ts_series: pd.Series) -> pd.Series:
    """PX4 datetime string → saniye (float)"""
    ts = pd.to_datetime(ts_series, errors="coerce")
    ts_int = ts.astype(np.int64)   # pandas 'ns' etiketli ama değerler PX4 µs
    return ts_int / 1e6            # µs → saniye


def load_flight(csv_path: str) -> pd.DataFrame | None:
    try:
        needed = [TIMESTAMP_COL] + INPUT_COLS + TARGET_COLS + REF_COLS
        all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        avail = [c for c in needed if c in all_cols]
        df = pd.read_csv(csv_path, usecols=avail, low_memory=False)
        return df
    except Exception as e:
        print(f"  Yukleme hatasi {Path(csv_path).name}: {e}")
        return None


def process_flight(df: pd.DataFrame, flight_id: str) -> dict | None:
    """Tek uçuşu işler, dict döndürür."""
    df = df.copy()

    # Timestamp saniyeye çevir
    if TIMESTAMP_COL in df.columns:
        df["time_s"] = parse_timestamp_s(df[TIMESTAMP_COL])
        df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    else:
        return None

    if len(df) < SEQUENCE_LENGTH + 5:
        return None

    # Girdi sütunları NaN doldur
    avail_inputs = [c for c in INPUT_COLS if c in df.columns]
    for col in avail_inputs:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(method="linear", limit=5).ffill().bfill().fillna(0.0)

    # Hedef sütunlar NaN doldur
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear", limit=10).ffill().bfill().fillna(0.0)

    # NED delta pozisyon
    # x_f58 = North (m), y_f58 = East (m), z_f58 = Down (m, negatif = yukarı)
    df["delta_north"] = df["x_f58"].diff().fillna(0.0) if "x_f58" in df.columns else 0.0
    df["delta_east"] = df["y_f58"].diff().fillna(0.0) if "y_f58" in df.columns else 0.0
    df["delta_up"] = (-df["z_f58"]).diff().fillna(0.0) if "z_f58" in df.columns else 0.0

    # Büyük sıçramaları filtrele (GPS reset artefaktı)
    for col in ["delta_north", "delta_east", "delta_up"]:
        q99 = df[col].abs().quantile(0.99)
        threshold = max(q99 * 3, 50.0)   # 50m/step makul üst sınır
        df.loc[df[col].abs() > threshold, col] = 0.0

    # Referans konum (NED → lat/lon dönüşümü için, modele verilmez)
    ref_lat = df["ref_lat_f58"].iloc[0] if "ref_lat_f58" in df.columns else np.nan
    ref_lon = df["ref_lon_f58"].iloc[0] if "ref_lon_f58" in df.columns else np.nan
    ref_alt = df["ref_alt_f58"].iloc[0] if "ref_alt_f58" in df.columns else np.nan

    # Feature sayısı: avail_inputs kadar (eksik sütunlar 0 doldurulacak)
    X_arr = np.zeros((len(df), len(INPUT_COLS)), dtype=np.float32)
    for j, col in enumerate(INPUT_COLS):
        if col in df.columns:
            X_arr[:, j] = df[col].values.astype(np.float32)

    y_delta = df[["delta_north", "delta_east", "delta_up"]].values.astype(np.float32)
    y_abs = df[TARGET_COLS].values.astype(np.float32) if all(c in df.columns for c in TARGET_COLS) else np.zeros((len(df), 3), dtype=np.float32)
    time_s = df["time_s"].values.astype(np.float32)

    return {
        "flight_id": flight_id,
        "X": X_arr,
        "y_delta": y_delta,
        "y_abs": y_abs,
        "time_s": time_s,
        "n_rows": len(df),
        "duration_s": float(time_s[-1] - time_s[0]),
        "ref_lat": float(ref_lat) if not np.isnan(ref_lat) else 0.0,
        "ref_lon": float(ref_lon) if not np.isnan(ref_lon) else 0.0,
        "ref_alt": float(ref_alt) if not np.isnan(ref_alt) else 0.0,
    }


def create_sequences(X: np.ndarray, y: np.ndarray,
                     seq_len: int = SEQUENCE_LENGTH,
                     step: int = STEP_SIZE):
    """Sliding window pencere dizileri."""
    X_seqs, y_seqs = [], []
    T = len(X)
    for i in range(0, T - seq_len, step):
        X_seqs.append(X[i:i + seq_len])
        y_seqs.append(y[i + seq_len - 1])
    if not X_seqs:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


def main():
    print("=" * 60)
    print("ADIM 3: On Isleme Pipeline")
    print("=" * 60)

    splits_path = OUT_DIR / "flight_splits.csv"
    if not splits_path.exists():
        print("HATA: flight_splits.csv bulunamadi. Once 02_quality_filter.py calistirin.")
        return

    splits_df = pd.read_csv(splits_path)
    print(f"Toplam gecerli ucus: {len(splits_df)}")
    print(f"Sequence: {SEQUENCE_LENGTH} adim = {SEQUENCE_LENGTH/ACTUAL_HZ:.0f} saniye")
    print(f"Step: {STEP_SIZE} adim = {STEP_SIZE/ACTUAL_HZ:.0f} saniye\n")

    split_data = {"train": [], "val": [], "test": []}
    flight_meta = []

    for split_name in ["train", "val", "test"]:
        subset = splits_df[splits_df["split"] == split_name]
        print(f"  {split_name.upper()} ({len(subset)} ucus):")

        for _, row in subset.iterrows():
            flight_id = row["flight_id"]
            df = load_flight(row["path"])
            if df is None:
                print(f"    [!!] {flight_id}: yuklenemedi")
                continue

            result = process_flight(df, flight_id)
            if result is None:
                print(f"    [!!] {flight_id}: isleme hatasi")
                continue

            split_data[split_name].append(result)
            flight_meta.append({
                "flight_id": flight_id,
                "split": split_name,
                "n_rows": result["n_rows"],
                "duration_s": result["duration_s"],
                "ref_lat": result["ref_lat"],
                "ref_lon": result["ref_lon"],
                "ref_alt": result["ref_alt"],
            })
            print(f"    [OK] {flight_id}: {result['n_rows']} satir, "
                  f"{result['duration_s']:.1f}s")

    print()

    # --- Normalizasyon ---
    print("  Normalizasyon (sadece train)...")
    if not split_data["train"]:
        print("HATA: Egitim seti bos.")
        return

    X_train_all = np.vstack([d["X"] for d in split_data["train"]])
    scaler = StandardScaler()
    scaler.fit(X_train_all)
    print(f"  Scaler fit: shape={X_train_all.shape}, "
          f"mean={scaler.mean_[:3].round(4)}")

    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # --- Pencere dizileri ---
    print("\n  Pencere dizileri olusturuluyor...")
    for split_name in ["train", "val", "test"]:
        X_seqs_list, y_seqs_list = [], []
        for d in split_data[split_name]:
            X_sc = scaler.transform(d["X"])
            X_seq, y_seq = create_sequences(X_sc, d["y_delta"])
            if len(X_seq) > 0:
                X_seqs_list.append(X_seq)
                y_seqs_list.append(y_seq)

        if X_seqs_list:
            X_final = np.vstack(X_seqs_list)
            y_final = np.vstack(y_seqs_list)
        else:
            X_final = np.empty((0, SEQUENCE_LENGTH, len(INPUT_COLS)), dtype=np.float32)
            y_final = np.empty((0, 3), dtype=np.float32)

        np.save(OUT_DIR / f"X_{split_name}.npy", X_final)
        np.save(OUT_DIR / f"y_{split_name}.npy", y_final)
        print(f"  {split_name}: X={X_final.shape}, y={y_final.shape}")

    # --- Ham uçuş verileri (GPS kesintisi deneyleri için) ---
    print("\n  Ham ucus verileri kaydediliyor...")
    for split_name in ["train", "val", "test"]:
        for d in split_data[split_name]:
            np.savez_compressed(
                OUT_DIR / f"flight_{d['flight_id']}_{split_name}.npz",
                X_raw=d["X"],
                X_scaled=scaler.transform(d["X"]),
                y_delta=d["y_delta"],
                y_abs=d["y_abs"],
                time_s=d["time_s"],
            )

    # Meta & özet
    meta_df = pd.DataFrame(flight_meta)
    meta_df.to_csv(OUT_DIR / "flight_meta.csv", index=False)

    summary = {
        "actual_hz": ACTUAL_HZ,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_seconds": SEQUENCE_LENGTH / ACTUAL_HZ,
        "step_size": STEP_SIZE,
        "n_features": len(INPUT_COLS),
        "input_cols": INPUT_COLS,
        "flights_processed": len(flight_meta),
        "split_sizes": {
            s: int(np.load(OUT_DIR / f"X_{s}.npy").shape[0])
            for s in ["train", "val", "test"]
        },
    }
    with open(OUT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("ON ISLEME TAMAMLANDI")
    print(f"  Train ornegi : {summary['split_sizes']['train']:,}")
    print(f"  Val ornegi   : {summary['split_sizes']['val']:,}")
    print(f"  Test ornegi  : {summary['split_sizes']['test']:,}")
    print(f"  Feature sayisi: {summary['n_features']}")
    print(f"  Pencere : {SEQUENCE_LENGTH} adim = {SEQUENCE_LENGTH/ACTUAL_HZ:.0f}s")
    print("\nAdim 3 tamamlandi.")
    return summary


if __name__ == "__main__":
    main()
