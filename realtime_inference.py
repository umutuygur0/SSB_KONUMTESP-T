"""
realtime_inference.py — GPS-Free UAV Konum Tahmini: Gerçek Zamanlı Demo

Kullanım:
    python realtime_inference.py --flight vuelo_100

Bir CSV uçuş dosyasını satır satır okur (gerçek zamanlı sensör akışı simülasyonu).
GPS verisini gizler; yalnızca IMU/mag/baro/airspeed ile konum tahmin eder.
Her 0.5s'de tahmini Enlem/Boylam/İrtifa ekrana yazar ve sonunda plot üretir.
"""
import sys, argparse, math, importlib.util, time
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# ── Yollar ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent
PH2     = BASE / "Phase2_Model" / "outputs"
SCRIPTS = BASE / "Phase2_Model" / "scripts"
RAW_DIR = BASE / "Holybro Pixhawk Dataset" / "Holybro Pixhawk" / "processed" / "Grouped flights"
OUT_DIR = BASE / "Phase4_Rapor" / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model yükle ─────────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location("models", SCRIPTS / "models.py")
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
detrend_single = _mod.detrend_single
GRUModel       = _mod.GRUModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru = GRUModel().to(device)
gru.load_state_dict(torch.load(PH2 / "best_gru.pt", weights_only=True, map_location=device))
gru.eval()

# ── Sabitler ────────────────────────────────────────────────────────────────
SEQ_LEN  = 40
FS       = 2       # Hz (0.5s/adım)
R_EARTH  = 6_371_000.0

INPUT_COLS = [
    "delta_angle[0]_f124", "delta_angle[1]_f124", "delta_angle[2]_f124",
    "delta_velocity[0]_f124", "delta_velocity[1]_f124", "delta_velocity[2]_f124",
    "mag_field[0]_f49", "mag_field[1]_f49", "mag_field[2]_f49",
    "indicated_airspeed_m_s_f5",
    "baro_alt_meter_f117",
    "differential_pressure_pa_f15",
]
GPS_TRUTH_COLS = ["x_f58", "y_f58", "z_f58"]  # NED ground truth (sadece doğrulama için)


def find_cols(df_cols, candidates):
    """Sütun adını farklı f-index varyantları arasında bul."""
    for c in candidates:
        if c in df_cols:
            return c
    base = candidates[0].rsplit("_f", 1)[0]
    for c in df_cols:
        if c.startswith(base + "_f"):
            return c
    return None


def ned_to_latlon(lat0, lon0, alt0, north_m, east_m, up_m):
    lat0_r = math.radians(lat0)
    lat = lat0 + (north_m / R_EARTH) * (180 / math.pi)
    lon = lon0 + (east_m  / (R_EARTH * math.cos(lat0_r))) * (180 / math.pi)
    alt = alt0 + up_m
    return lat, lon, alt


def run_realtime(flight_name: str, verbose: bool = True):
    csv_path = RAW_DIR / f"{flight_name}_sync.csv"
    if not csv_path.exists():
        print(f"HATA: {csv_path} bulunamadı."); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  GPS-Free Gerçek Zamanlı Inference Demo")
    print(f"  Uçuş: {flight_name}  |  Device: {device}")
    print(f"{'='*60}\n")

    df = pd.read_csv(csv_path)

    # Referans GPS koordinatları (başlangıç noktası)
    ref_lat = float(df["ref_lat_f58"].dropna().iloc[0])
    ref_lon = float(df["ref_lon_f58"].dropna().iloc[0])
    ref_alt = float(df["ref_alt_f58"].dropna().iloc[0])
    print(f"  Referans konum: {ref_lat:.6f}°N, {ref_lon:.6f}°E, {ref_alt:.1f}m MSL\n")

    # Girdi sütunlarını bul
    actual_input_cols = []
    for col in INPUT_COLS:
        found = find_cols(df.columns.tolist(), [col])
        if found is None:
            print(f"  UYARI: {col} bulunamadı, 0 kullanılacak.")
        actual_input_cols.append(found)

    # GPS ground truth (doğrulama)
    truth_x = df["x_f58"].values if "x_f58" in df.columns else None
    truth_y = df["y_f58"].values if "y_f58" in df.columns else None

    # Scaler istatistikleri (Phase1'den fit edilmiş) — basit min-max norm yerine
    # X_scaled zaten kaydedilmiş, ham veriyi ölçeklendirmek için scaler.pkl kullanırız
    from pathlib import Path as _P
    import pickle
    scaler_path = BASE / "Phase1_Veri_Analizi" / "outputs" / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        use_scaler = True
    else:
        use_scaler = False
        print("  UYARI: scaler.pkl bulunamadı, ham değerler kullanılıyor.")

    # Sliding window buffer
    window_buf = deque(maxlen=SEQ_LEN)

    # Kümülatif konum
    cum_north = 0.0
    cum_east  = 0.0
    cum_up    = 0.0

    # Kayıt listeleri
    pred_lats, pred_lons, pred_alts = [], [], []
    true_norths, true_easts = [], []
    hpe_list = []
    step_times = []

    print(f"  {'Adım':>5}  {'Zaman(s)':>9}  {'Enlem(°)':>12}  {'Boylam(°)':>12}  {'İrtifa(m)':>10}  {'HPE(m)':>8}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")

    N = len(df)
    for step in range(N):
        row = df.iloc[step]
        t_s = float(step) / FS

        # Ham girdi vektörü
        raw_vec = np.zeros(12, dtype=np.float32)
        for i, col in enumerate(actual_input_cols):
            if col is not None and col in df.columns:
                v = row[col]
                raw_vec[i] = float(v) if not pd.isna(v) else 0.0

        # Ölçeklendir
        if use_scaler:
            raw_vec_s = scaler.transform(raw_vec.reshape(1, -1))[0].astype(np.float32)
        else:
            raw_vec_s = raw_vec

        window_buf.append(raw_vec_s)

        # Pencere dolmadan tahmin yapma
        if len(window_buf) < SEQ_LEN:
            continue

        # Per-window detrend + inference
        win_arr = np.array(window_buf, dtype=np.float32)  # (SEQ_LEN, 12)
        win_dt  = detrend_single(win_arr)                  # per-window z-score
        inp = torch.from_numpy(win_dt).unsqueeze(0).to(device)  # (1, SEQ_LEN, 12)

        with torch.no_grad():
            pred = gru(inp).cpu().numpy()[0]   # [ΔNorth, ΔEast, ΔUp]

        cum_north += pred[0]
        cum_east  += pred[1]
        cum_up    += pred[2]

        lat, lon, alt = ned_to_latlon(ref_lat, ref_lon, ref_alt, cum_north, cum_east, cum_up)
        pred_lats.append(lat); pred_lons.append(lon); pred_alts.append(alt)
        step_times.append(t_s)

        # Ground truth HPE
        hpe = None
        if truth_x is not None and step < len(truth_x):
            true_n = float(truth_x[step]) if not np.isnan(truth_x[step]) else 0.0
            true_e = float(truth_y[step]) if not np.isnan(truth_y[step]) else 0.0
            true_norths.append(true_n); true_easts.append(true_e)
            hpe = math.sqrt((cum_north - true_n)**2 + (cum_east - true_e)**2)
            hpe_list.append(hpe)

        if verbose and (step % 20 == 0 or step == N - 1):
            hpe_str = f"{hpe:8.2f}" if hpe is not None else "       —"
            print(f"  {step:>5}  {t_s:>9.1f}  {lat:>12.6f}  {lon:>12.6f}  {alt:>10.1f}  {hpe_str}")

    # ── Sonuç özeti ──────────────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  SONUÇ (uçuş sonu tahmini):")
    print(f"    Tahmin   : {pred_lats[-1]:.6f}°N  {pred_lons[-1]:.6f}°E  {pred_alts[-1]:.1f}m")
    if hpe_list:
        print(f"    HPE final: {hpe_list[-1]:.2f}m")
        print(f"    HPE ort.  : {np.mean(hpe_list):.2f}m")
        print(f"    HPE p95   : {np.percentile(hpe_list,95):.2f}m")
    print(f"    Adım sayısı: {len(pred_lats)}  |  Uçuş süresi: {step_times[-1]:.1f}s")

    # ── Grafik ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Gerçek Zamanlı Inference — {flight_name}\nGPS Yok: Yalnızca IMU/Mag/Baro/Airspeed",
                 fontsize=12, fontweight="bold")

    # Rota
    ax = axes[0]
    if true_norths:
        ax.plot(true_easts[:len(pred_lats)], true_norths[:len(pred_lats)],
                "b-", lw=2, label="GPS Gerçeği", alpha=0.8)
    ax.plot([lon - ref_lon for lon in pred_lons],
            [lat - ref_lat for lat in pred_lats],
            "r--", lw=1.5, label="GRU Tahmin", alpha=0.8)
    ax.set_xlabel("ΔBoylam (°)"); ax.set_ylabel("ΔEnlem (°)")
    ax.set_title("2D Rota (Enlem-Boylam farkı)")
    ax.legend(); ax.grid(alpha=0.3)

    # İrtifa
    ax2 = axes[1]
    ax2.plot(step_times, pred_alts, "r-", lw=1.5, label="GRU İrtifa (m)")
    ax2.axhline(ref_alt, color="gray", linestyle=":", label=f"Ref Alt={ref_alt:.0f}m")
    ax2.set_xlabel("Zaman (s)"); ax2.set_ylabel("İrtifa (m)")
    ax2.set_title("İrtifa Tahmini vs Zaman")
    ax2.legend(); ax2.grid(alpha=0.3)

    # HPE
    ax3 = axes[2]
    if hpe_list:
        ax3.plot(step_times[:len(hpe_list)], hpe_list, "purple", lw=1.5)
        ax3.fill_between(step_times[:len(hpe_list)], 0, hpe_list, alpha=0.15, color="purple")
        ax3.axhline(np.mean(hpe_list), color="red", linestyle="--",
                    label=f"Ort={np.mean(hpe_list):.1f}m")
        ax3.set_xlabel("Zaman (s)"); ax3.set_ylabel("HPE (m)")
        ax3.set_title("Kümülatif HPE")
        ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / f"realtime_demo_{flight_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grafik: {out_png}")

    return {
        "flight": flight_name,
        "n_steps": len(pred_lats),
        "flight_dur_s": float(step_times[-1]),
        "final_lat": round(float(pred_lats[-1]), 7),
        "final_lon": round(float(pred_lons[-1]), 7),
        "final_alt_m": round(float(pred_alts[-1]), 2),
        "HPE_mean": round(float(np.mean(hpe_list)), 2) if hpe_list else None,
        "HPE_p95":  round(float(np.percentile(hpe_list, 95)), 2) if hpe_list else None,
        "HPE_final": round(float(hpe_list[-1]), 2) if hpe_list else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPS-Free Gerçek Zamanlı Inference Demo")
    parser.add_argument("--flight", default="vuelo_100",
                        help="Uçuş adı (ör: vuelo_100)")
    parser.add_argument("--quiet", action="store_true",
                        help="Satır satır çıktıyı kapat")
    args = parser.parse_args()

    result = run_realtime(args.flight, verbose=not args.quiet)
    print("\n  İşlem tamamlandı.")
