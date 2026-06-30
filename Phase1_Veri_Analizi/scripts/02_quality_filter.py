"""
Phase 1 — Adım 2: Uçuş Kalite Filtresi (Düzeltilmiş)

Veri formatı notları:
- Grouped flights CSV: ~2 Hz senkronize zaman serisi, ~500 satır/uçuş (~250 saniye)
- Timestamp: PX4 microsaniye boot zamanı, datetime string olarak kaydedilmiş
  Örnek: '1970-01-01 00:00:00.148570132' → 148570132 µs from boot = 148.57 saniye
  pandas.to_datetime().astype(int64) değeri [ns] olarak okur, ama GERÇEKTE µs'dir.
  Bu yüzden ts.astype(int64) / 1e6 = saniye (µs/1e6 = s)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
GROUPED_DIR = BASE_DIR / "Holybro Pixhawk Dataset" / "Holybro Pixhawk" / "processed" / "Grouped flights"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Kalite kriterleri
MIN_ROWS = 100           # ~50 saniye @ ~2 Hz
MIN_DURATION_S = 30.0   # Minimum uçuş süresi
MAX_NAN_RATIO = 0.50    # Kritik sütunlarda max %50 boş değer (sparse formatta tolerans)
TIMESTAMP_COL = "timestamp"

CRITICAL_INPUT_COLS = [
    "delta_angle[0]_f124", "delta_velocity[0]_f124",
    "mag_field[0]_f49", "indicated_airspeed_m_s_f5",
    "baro_alt_meter_f117",
]
CRITICAL_TARGET_COLS = ["x_f58", "y_f58", "z_f58"]


def parse_timestamp_s(ts_series: pd.Series) -> pd.Series:
    """
    PX4 timestamp'ini saniyeye çevirir.
    Pandas datetime olarak okur, int64 (ns olarak etiketli ama gerçekte µs) alır,
    1e6'ya bölerek saniye elde eder.
    """
    ts = pd.to_datetime(ts_series, errors="coerce")
    ts_int = ts.astype(np.int64)   # ns olarak etiketli ama gerçekte µs değerleri
    ts_s = ts_int / 1e6            # µs → s
    return ts_s


def check_flight(csv_path: Path) -> dict:
    result = {
        "file": csv_path.name,
        "flight_id": csv_path.stem.replace("_sync", ""),
        "path": str(csv_path),
        "status": "unknown",
        "reject_reason": "",
        "row_count": 0,
        "duration_s": 0.0,
        "sample_hz": 0.0,
        "file_size_mb": round(csv_path.stat().st_size / 1e6, 2),
    }

    try:
        # Header oku
        df_head = pd.read_csv(csv_path, nrows=0)
        all_cols = set(df_head.columns.tolist())

        # Kritik sütun varlığı
        missing_inputs = [c for c in CRITICAL_INPUT_COLS if c not in all_cols]
        missing_targets = [c for c in CRITICAL_TARGET_COLS if c not in all_cols]
        if missing_inputs or missing_targets:
            result["status"] = "rejected"
            result["reject_reason"] = f"Eksik kritik sutun: {(missing_inputs + missing_targets)[:3]}"
            return result

        # Veri oku (gerekli sütunlar)
        needed = [TIMESTAMP_COL] + CRITICAL_INPUT_COLS + CRITICAL_TARGET_COLS
        needed = [c for c in needed if c in all_cols]
        df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
        result["row_count"] = len(df)

        # Minimum satır
        if len(df) < MIN_ROWS:
            result["status"] = "rejected"
            result["reject_reason"] = f"Az satir: {len(df)} < {MIN_ROWS}"
            return result

        # Uçuş süresi
        if TIMESTAMP_COL in df.columns:
            ts_s = parse_timestamp_s(df[TIMESTAMP_COL]).dropna()
            if len(ts_s) > 1:
                duration_s = ts_s.iloc[-1] - ts_s.iloc[0]
                dt_mean = ts_s.diff().dropna().mean()
                result["duration_s"] = round(float(duration_s), 2)
                result["sample_hz"] = round(float(1.0 / dt_mean) if dt_mean > 0 else 0, 2)
                if duration_s < MIN_DURATION_S:
                    result["status"] = "rejected"
                    result["reject_reason"] = f"Kisa ucus: {duration_s:.1f}s < {MIN_DURATION_S}s"
                    return result
            else:
                result["status"] = "rejected"
                result["reject_reason"] = "Timestamp parse edilemedi"
                return result

        # NaN oranı
        for col in CRITICAL_INPUT_COLS + CRITICAL_TARGET_COLS:
            if col in df.columns:
                nan_ratio = float(df[col].isna().mean())
                if nan_ratio > MAX_NAN_RATIO:
                    result["status"] = "rejected"
                    result["reject_reason"] = f"Yuksek NaN: {col[:20]} -> {nan_ratio:.0%}"
                    return result
                result[f"nan_{col.split('[')[0][:15]}"] = round(nan_ratio, 4)

        # Timestamp monotonluk
        if TIMESTAMP_COL in df.columns:
            ts_s = parse_timestamp_s(df[TIMESTAMP_COL]).dropna()
            diffs = ts_s.diff().dropna()
            neg_count = int((diffs < 0).sum())
            if neg_count > 5:
                result["status"] = "warning"
                result["reject_reason"] = f"Timestamp monoton degil ({neg_count} ters adim, uyari)"
            else:
                result["status"] = "ok"
        else:
            result["status"] = "ok"

    except Exception as e:
        result["status"] = "error"
        result["reject_reason"] = str(e)[:80]

    return result


def main():
    print("=" * 60)
    print("ADIM 2: Ucus Kalite Filtresi")
    print("=" * 60)

    csv_files = sorted(GROUPED_DIR.glob("vuelo_*_sync.csv"))
    print(f"Toplam dosya: {len(csv_files)}")
    print(f"Kalite kriterleri: min_rows={MIN_ROWS}, min_dur={MIN_DURATION_S}s\n")

    results = []
    for i, f in enumerate(csv_files, 1):
        r = check_flight(f)
        results.append(r)
        icon = {"ok": "[OK]", "warning": "[!!]", "rejected": "[--]", "error": "[ER]"}.get(r["status"], "[?]")
        print(f"  [{i:3d}/{len(csv_files)}] {icon} {r['file']:30s} | {r['row_count']:4d} satir | "
              f"{r['duration_s']:7.1f}s | {r['sample_hz']:5.1f}Hz | {r['reject_reason'][:40]}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "quality_report.csv", index=False)

    ok_flights = df_results[df_results["status"].isin(["ok", "warning"])]
    rejected = df_results[df_results["status"] == "rejected"]
    errors = df_results[df_results["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"OZET")
    print(f"  Kullanilabilir: {len(ok_flights)}")
    print(f"  Reddedilen    : {len(rejected)}")
    print(f"  Hata          : {len(errors)}")

    if len(rejected) > 0:
        print("\nReddedilen ucuslar:")
        for _, row in rejected.iterrows():
            print(f"  {row['file']:30s} -> {row['reject_reason']}")

    # Geçerli uçuş listesi — uçuş numarasına göre sıralı
    valid_list = ok_flights[["file", "flight_id", "path", "row_count", "duration_s", "sample_hz"]].copy()
    valid_list["_num"] = valid_list["flight_id"].str.extract(r"vuelo_(\d+)").astype(int)
    valid_list = valid_list.sort_values("_num").drop("_num", axis=1).reset_index(drop=True)
    valid_list.to_csv(OUT_DIR / "valid_flights.csv", index=False)

    # Split — uçuş numarası sırası korunarak
    n = len(valid_list)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    valid_list["split"] = "test"
    valid_list.loc[:n_train - 1, "split"] = "train"
    valid_list.loc[n_train:n_train + n_val - 1, "split"] = "val"
    valid_list.to_csv(OUT_DIR / "flight_splits.csv", index=False)
    print(f"\nSplit: Train={n_train} | Val={n_val} | Test={n_test}")

    # İstatistikler
    if len(ok_flights) > 0:
        dur_vals = ok_flights["duration_s"]
        hz_vals = ok_flights["sample_hz"]
        print(f"\nUcus suresi istatistikleri:")
        print(f"  Ortalama: {dur_vals.mean():.1f}s | Medyan: {dur_vals.median():.1f}s")
        print(f"  Min: {dur_vals.min():.1f}s | Max: {dur_vals.max():.1f}s")
        print(f"  Toplam: {dur_vals.sum()/60:.1f} dakika")
        print(f"  Ortalama Hz: {hz_vals.mean():.2f}")

    summary = {
        "total_files": len(csv_files),
        "valid_flights": int(len(ok_flights)),
        "rejected_flights": int(len(rejected)),
        "error_flights": int(len(errors)),
        "split": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "rejected_list": rejected["file"].tolist(),
        "avg_duration_s": float(ok_flights["duration_s"].mean()) if len(ok_flights) > 0 else 0,
        "avg_hz": float(ok_flights["sample_hz"].mean()) if len(ok_flights) > 0 else 0,
    }
    with open(OUT_DIR / "quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nAdim 2 tamamlandi.")
    return summary


if __name__ == "__main__":
    main()
