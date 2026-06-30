"""
Phase 1 — Adım 1: Sütun Keşfi ve F-İndeks Katalogu
Holybro Grouped CSV'deki 1046 sütunu analiz eder, f-indekslerini gruplar,
sensör tiplerini tespit eder ve bir katalog CSV'si üretir.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# --- Yollar ---
BASE_DIR = Path(__file__).resolve().parents[2]
GROUPED_DIR = BASE_DIR / "Holybro Pixhawk Dataset" / "Holybro Pixhawk" / "processed" / "Grouped flights"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_FILE = GROUPED_DIR / "vuelo_1_sync.csv"

# --- Sensör pattern'leri ---
SENSOR_PATTERNS = {
    "delta_angle":          r"^delta_angle\[\d\]",
    "delta_velocity":       r"^delta_velocity\[\d\]",
    "delta_dt":             r"^delta_(angle|velocity)_dt",
    "angular_velocity":     r"^angular_velocity\[\d\]",
    "mag_field":            r"^mag_field\[\d\]",
    "airspeed":             r"^(indicated|true|calibrated)_airspeed",
    "baro_alt":             r"^baro_alt_meter",
    "diff_pressure":        r"^differential_pressure",
    "quaternion":           r"^q\[\d\]",
    "gps_lat_lon":          r"^(lat|lon)_f",
    "gps_alt":              r"^alt_f",
    "ned_position":         r"^[xyz]_f\d+$",
    "ned_velocity":         r"^[vw][xyz]_f\d+$",
    "ned_accel":            r"^a[xyz]_f\d+$",
    "ref_position":         r"^ref_(lat|lon|alt)",
    "wind":                 r"^windspeed_(north|east)",
    "battery":              r"^(voltage|current|remaining)",
    "temperature":          r"^temperature",
    "baro_vpos":            r"^baro_vpos",
    "control_output":       r"^(control|output)\[",
    "actuator":             r"^(noutputs|output\[)",
}

GPS_LEAKAGE_PATTERNS = [
    r"^(lat|lon)_f",
    r"^alt_f\d+$",
    r"^[xyz]_f\d+$",
    r"^[vw][xyz]_f\d+$",
    r"^a[xyz]_f\d+$",
    r"^ref_(lat|lon|alt)",
    r"^windspeed_(north|east)",
    r"^lat_lon_valid",
    r"^lat_lon_reset",
    r"^terrain_alt",
    r"^delta_alt",
    r"^gnss",
    r"^(gps|GNSS)",
    r"yaw_aligned_to_imu_gps",
    r"^estimator.*pos",
    r"^vehicle.*position",
]


def classify_column(col_name: str) -> tuple[str, bool]:
    """Sütunun sensör tipini ve GPS leakage riskini döndürür."""
    import re
    sensor_type = "unknown"
    for name, pattern in SENSOR_PATTERNS.items():
        if re.search(pattern, col_name):
            sensor_type = name
            break

    is_leakage = any(re.search(p, col_name, re.IGNORECASE) for p in GPS_LEAKAGE_PATTERNS)
    return sensor_type, is_leakage


def extract_f_index(col_name: str) -> str | None:
    """Sütun adından f-indeksini çıkarır. Örn: 'delta_angle[0]_f124' → 'f124'"""
    import re
    m = re.search(r"_f(\d+)$", col_name)
    return f"f{m.group(1)}" if m else None


def analyze_columns(csv_path: Path) -> pd.DataFrame:
    print(f"Okuma: {csv_path.name}")
    # Sadece header + 2 veri satırı oku (hız için)
    df_head = pd.read_csv(csv_path, nrows=2)
    cols = df_head.columns.tolist()
    print(f"  Toplam sütun: {len(cols)}")

    records = []
    for col in cols:
        f_idx = extract_f_index(col)
        sensor_type, is_leakage = classify_column(col)
        records.append({
            "column": col,
            "f_index": f_idx,
            "sensor_type": sensor_type,
            "gps_leakage_risk": is_leakage,
            "sample_val_row1": df_head[col].iloc[0] if len(df_head) > 0 else None,
            "sample_val_row2": df_head[col].iloc[1] if len(df_head) > 1 else None,
        })

    return pd.DataFrame(records)


def summarize_f_groups(catalog: pd.DataFrame) -> pd.DataFrame:
    """Her f-indeks grubunun özet istatistiklerini çıkarır."""
    rows = []
    for f_idx, grp in catalog.groupby("f_index"):
        sensor_types = grp["sensor_type"].value_counts().to_dict()
        rows.append({
            "f_index": f_idx,
            "col_count": len(grp),
            "sensor_types": str(sensor_types),
            "has_leakage": grp["gps_leakage_risk"].any(),
            "sample_cols": "|".join(grp["column"].head(3).tolist()),
        })
    return pd.DataFrame(rows).sort_values("f_index")


def main():
    print("=" * 60)
    print("ADIM 1: Sütun Keşfi ve Kataloglama")
    print("=" * 60)

    catalog = analyze_columns(SAMPLE_FILE)

    # Kaydet
    catalog_path = OUT_DIR / "column_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    print(f"\nKatalog kaydedildi: {catalog_path}")

    # F-grup özeti
    f_summary = summarize_f_groups(catalog)
    f_summary_path = OUT_DIR / "f_index_summary.csv"
    f_summary.to_csv(f_summary_path, index=False)
    print(f"F-indeks özeti kaydedildi: {f_summary_path}")

    # Önerilen input sütunları
    print("\n--- ÖNERİLEN GİRDİ SÜTUNLARI ---")
    INPUT_CANDIDATES = [
        "delta_angle[0]_f124", "delta_angle[1]_f124", "delta_angle[2]_f124",
        "delta_angle_dt_f124",
        "delta_velocity[0]_f124", "delta_velocity[1]_f124", "delta_velocity[2]_f124",
        "delta_velocity_dt_f124",
        "mag_field[0]_f49", "mag_field[1]_f49", "mag_field[2]_f49",
        "indicated_airspeed_m_s_f5",
        "baro_alt_meter_f117",
        "differential_pressure_pa_f15",
    ]
    TARGET_CANDIDATES = ["x_f58", "y_f58", "z_f58", "ref_lat_f58", "ref_lon_f58", "ref_alt_f58"]

    all_cols = set(catalog["column"].tolist())
    found_inputs = [c for c in INPUT_CANDIDATES if c in all_cols]
    missing_inputs = [c for c in INPUT_CANDIDATES if c not in all_cols]
    found_targets = [c for c in TARGET_CANDIDATES if c in all_cols]

    print(f"Girdi sutunlari bulundu ({len(found_inputs)}/{len(INPUT_CANDIDATES)}):")
    for c in found_inputs:
        print(f"  [OK] {c}")
    if missing_inputs:
        print(f"\nEKSIK SUTUNLAR ({len(missing_inputs)}):")
        for c in missing_inputs:
            print(f"  [!!] {c}")

    print(f"\nHedef sutunlar bulundu ({len(found_targets)}/{len(TARGET_CANDIDATES)}):")
    for c in found_targets:
        print(f"  [OK] {c}")

    # GPS leakage özeti
    leakage_cols = catalog[catalog["gps_leakage_risk"]]["column"].tolist()
    print(f"\nGPS Leakage Riski Tespit Edilen Sütun Sayısı: {len(leakage_cols)}")

    # Sonuç JSON
    result = {
        "total_columns": len(catalog),
        "found_inputs": found_inputs,
        "missing_inputs": missing_inputs,
        "found_targets": found_targets,
        "gps_leakage_count": len(leakage_cols),
        "unique_f_indices": catalog["f_index"].dropna().nunique(),
    }
    with open(OUT_DIR / "column_exploration_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nToplam f-indeks grubu: {result['unique_f_indices']}")
    print("\nAdım 1 tamamlandı.")
    return result


if __name__ == "__main__":
    main()
