"""
Phase 1 — Adım 4: EDA Görselleştirme ve İstatistik Raporu
- Girdi sensör dağılımları
- Hedef NED konum dağılımları
- Örnek uçuş rotası (GPS track)
- Uçuş süresi dağılımı
- Eksik veri ısı haritası
- Tüm çıktıları outputs/ klasörüne kaydeder
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI olmadan çalışır
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
GROUPED_DIR = BASE_DIR / "Holybro Pixhawk Dataset" / "Holybro Pixhawk" / "processed" / "Grouped flights"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

INPUT_COLS = [
    "delta_angle[0]_f124", "delta_angle[1]_f124", "delta_angle[2]_f124",
    "delta_velocity[0]_f124", "delta_velocity[1]_f124", "delta_velocity[2]_f124",
    "mag_field[0]_f49", "mag_field[1]_f49", "mag_field[2]_f49",
    "indicated_airspeed_m_s_f5",
    "baro_alt_meter_f117",
    "differential_pressure_pa_f15",
]
TARGET_COLS = ["x_f58", "y_f58", "z_f58"]
TIMESTAMP_COL = "timestamp"
SAMPLE_FLIGHTS = ["vuelo_1_sync.csv", "vuelo_10_sync.csv", "vuelo_50_sync.csv"]


def load_sample(flight_name: str, cols: list) -> pd.DataFrame | None:
    path = GROUPED_DIR / flight_name
    if not path.exists():
        return None
    try:
        all_cols = pd.read_csv(path, nrows=0).columns.tolist()
        avail = [c for c in cols if c in all_cols]
        df = pd.read_csv(path, usecols=avail, low_memory=False)
        return df
    except Exception:
        return None


def plot_sensor_distributions(flights_data: dict):
    """Girdi sensör dağılımı (violin + boxplot)."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle("Girdi Sensör Dağılımları (3 örnek uçuş)", fontsize=14, fontweight="bold")

    labels = [
        "ΔAngle X", "ΔAngle Y", "ΔAngle Z",
        "ΔVel X", "ΔVel Y", "ΔVel Z",
        "Mag X", "Mag Y", "Mag Z",
        "Airspeed", "Baro Alt", "Diff Press"
    ]

    for idx, (col, label) in enumerate(zip(INPUT_COLS, labels)):
        ax = axes[idx // 3][idx % 3]
        data_parts = []
        flight_names = []
        for fname, df in flights_data.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    data_parts.append(vals.values[:5000])   # max 5000 örnek
                    flight_names.append(fname[:10])

        if data_parts:
            ax.boxplot(data_parts, labels=flight_names, vert=True, patch_artist=True,
                      medianprops=dict(color="red", linewidth=2))
            ax.set_title(label, fontsize=9)
            ax.set_ylabel("Değer", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
        else:
            ax.set_title(f"{label} — VERİ YOK", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_sensor_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_sensor_distributions.png")


def plot_flight_tracks(flights_data: dict):
    """GPS track ve NED konum karşılaştırması."""
    fig, axes = plt.subplots(1, len(flights_data), figsize=(6 * len(flights_data), 5))
    if len(flights_data) == 1:
        axes = [axes]
    fig.suptitle("Uçuş Rotaları (NED Yerel Konum)", fontsize=13, fontweight="bold")

    for ax, (fname, df) in zip(axes, flights_data.items()):
        x_col = "x_f58"   # North
        y_col = "y_f58"   # East
        if x_col in df.columns and y_col in df.columns:
            north = pd.to_numeric(df[x_col], errors="coerce").dropna()
            east = pd.to_numeric(df[y_col], errors="coerce").dropna()
            min_len = min(len(north), len(east))
            t = np.linspace(0, 1, min_len)
            sc = ax.scatter(east[:min_len], north[:min_len], c=t, cmap="plasma", s=1, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="Zaman (normalize)")
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")
            ax.set_title(fname[:20])
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "NED verisi yok", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_flight_tracks.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_flight_tracks.png")


def plot_altitude_profile(flights_data: dict):
    """İrtifa profili (baro vs NED Down)."""
    fig, axes = plt.subplots(len(flights_data), 1, figsize=(14, 4 * len(flights_data)), squeeze=False)
    fig.suptitle("İrtifa Profili — Baro vs NED-Down", fontsize=13, fontweight="bold")

    for i, (fname, df) in enumerate(flights_data.items()):
        ax = axes[i][0]
        n_pts = min(len(df), 5000)
        t = np.arange(n_pts)

        if "baro_alt_meter_f117" in df.columns:
            baro = pd.to_numeric(df["baro_alt_meter_f117"], errors="coerce").iloc[:n_pts]
            ax.plot(t, baro, label="Baro Alt (m)", color="blue", alpha=0.8, lw=0.8)

        if "z_f58" in df.columns:
            ned_z = pd.to_numeric(df["z_f58"], errors="coerce").iloc[:n_pts]
            ax.plot(t, -ned_z, label="NED-Up (m)", color="red", alpha=0.8, lw=0.8, linestyle="--")

        ax.set_title(fname[:30])
        ax.set_xlabel("Zaman adımı")
        ax.set_ylabel("İrtifa (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_altitude_profiles.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_altitude_profiles.png")


def plot_quality_overview():
    """Tüm uçuşlar için kalite özet grafiği."""
    quality_path = OUT_DIR / "quality_report.csv"
    if not quality_path.exists():
        print("  quality_report.csv yok, grafik atlanıyor")
        return

    df = pd.read_csv(quality_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Uçuş Kalite Genel Bakış", fontsize=13, fontweight="bold")

    # 1. Durum dağılımı (pasta)
    status_counts = df["status"].value_counts()
    colors = {"ok": "#2ecc71", "warning": "#f39c12", "rejected": "#e74c3c", "error": "#95a5a6"}
    pie_colors = [colors.get(s, "gray") for s in status_counts.index]
    axes[0].pie(status_counts.values, labels=status_counts.index, colors=pie_colors,
               autopct="%1.0f%%", startangle=90)
    axes[0].set_title("Uçuş Durumu Dağılımı")

    # 2. Dosya boyutu dağılımı
    ok_df = df[df["status"].isin(["ok", "warning"])]
    if len(ok_df) > 0:
        axes[1].hist(ok_df["file_size_mb"], bins=20, color="#3498db", alpha=0.8, edgecolor="white")
        axes[1].axvline(ok_df["file_size_mb"].median(), color="red", linestyle="--",
                       label=f"Medyan: {ok_df['file_size_mb'].median():.1f} MB")
        axes[1].set_xlabel("Dosya Boyutu (MB)")
        axes[1].set_ylabel("Uçuş Sayısı")
        axes[1].set_title("Dosya Boyutu Dağılımı")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # 3. Uçuş süresi dağılımı
    ok_df2 = df[(df["status"].isin(["ok", "warning"])) & (df["duration_s"] > 0)]
    if len(ok_df2) > 0:
        axes[2].hist(ok_df2["duration_s"], bins=20, color="#9b59b6", alpha=0.8, edgecolor="white")
        axes[2].axvline(ok_df2["duration_s"].median(), color="red", linestyle="--",
                       label=f"Medyan: {ok_df2['duration_s'].median():.0f}s")
        axes[2].set_xlabel("Uçuş Süresi (saniye)")
        axes[2].set_ylabel("Uçuş Sayısı")
        axes[2].set_title("Uçuş Süresi Dağılımı")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_quality_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_quality_overview.png")


def plot_nan_heatmap(flights_data: dict):
    """Eksik değer ısı haritası."""
    nan_records = {}
    check_cols = INPUT_COLS + TARGET_COLS

    for fname, df in flights_data.items():
        nan_records[fname] = {}
        for col in check_cols:
            if col in df.columns:
                nan_records[fname][col] = df[col].isna().mean()
            else:
                nan_records[fname][col] = 1.0  # Mevcut değil → %100 eksik

    nan_df = pd.DataFrame(nan_records).T

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(nan_df.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="NaN Oranı")
    ax.set_xticks(range(len(nan_df.columns)))
    ax.set_xticklabels([c[:20] for c in nan_df.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(nan_df.index)))
    ax.set_yticklabels(nan_df.index, fontsize=8)
    ax.set_title("Eksik Değer (NaN) Isı Haritası — Örnek Uçuşlar", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_nan_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_nan_heatmap.png")


def plot_imu_timeseries(flights_data: dict):
    """IMU zaman serisi — delta_angle ve delta_velocity."""
    fname, df = next(iter(flights_data.items()))
    n_pts = min(len(df), 2000)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"IMU Zaman Serisi — {fname[:25]}", fontsize=13, fontweight="bold")

    # Delta angle (gyro)
    ax = axes[0]
    for i, lbl in enumerate(["X", "Y", "Z"]):
        col = f"delta_angle[{i}]_f124"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").iloc[:n_pts]
            ax.plot(vals.values, label=f"ΔAngle {lbl}", alpha=0.8, lw=0.7)
    ax.set_title("Delta Angle (Jiroskop Entegrasyonu)")
    ax.set_ylabel("rad")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Delta velocity (accel)
    ax = axes[1]
    for i, lbl in enumerate(["X", "Y", "Z"]):
        col = f"delta_velocity[{i}]_f124"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").iloc[:n_pts]
            ax.plot(vals.values, label=f"ΔVel {lbl}", alpha=0.8, lw=0.7)
    ax.set_title("Delta Velocity (İvmeölçer Entegrasyonu)")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Satır (≈ örnekleme indeksi)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_imu_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ 06_imu_timeseries.png")


def compute_eda_stats(flights_data: dict) -> dict:
    """Temel EDA istatistikleri."""
    stats = {}
    all_cols = INPUT_COLS + TARGET_COLS
    combined = {col: [] for col in all_cols}

    for fname, df in flights_data.items():
        for col in all_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().values[:5000]
                combined[col].extend(vals.tolist())

    for col in all_cols:
        arr = np.array(combined[col])
        if len(arr) > 0:
            stats[col] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "p1": float(np.percentile(arr, 1)),
                "p99": float(np.percentile(arr, 99)),
            }
        else:
            stats[col] = {"error": "veri yok"}

    return stats


def main():
    print("=" * 60)
    print("ADIM 4: EDA Görselleştirme")
    print("=" * 60)

    # Örnek uçuşları yükle
    print("\n  Örnek uçuşlar yükleniyor...")
    all_cols_needed = [TIMESTAMP_COL] + INPUT_COLS + TARGET_COLS
    flights_data = {}
    for fname in SAMPLE_FLIGHTS:
        df = load_sample(fname, all_cols_needed)
        if df is not None:
            flights_data[fname.replace("_sync.csv", "")] = df
            print(f"  ✓ {fname}: {len(df)} satır")
        else:
            print(f"  ✗ {fname}: yüklenemedi")

    if not flights_data:
        print("HATA: Hiçbir örnek uçuş yüklenemedi.")
        return

    print("\n  Grafikler oluşturuluyor...")
    plot_sensor_distributions(flights_data)
    plot_flight_tracks(flights_data)
    plot_altitude_profile(flights_data)
    plot_quality_overview()
    plot_nan_heatmap(flights_data)
    plot_imu_timeseries(flights_data)

    # İstatistik raporu
    print("\n  EDA istatistikleri hesaplanıyor...")
    stats = compute_eda_stats(flights_data)
    with open(OUT_DIR / "eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ eda_stats.json kaydedildi")

    # İstatistik tablosu
    stat_rows = []
    for col, s in stats.items():
        if "error" not in s:
            stat_rows.append({"sütun": col, **s})
    pd.DataFrame(stat_rows).to_csv(OUT_DIR / "eda_stats_table.csv", index=False)

    print(f"\nTüm grafikler: {PLOTS_DIR}")
    print("\nAdım 4 tamamlandı.")


if __name__ == "__main__":
    main()
