"""
03_ned_to_latlon_demo.py
NED (metre) → Enlem/Boylam/İrtifa (°, m) dönüşümü demonstrasyonu.
GRU model çıktısı (ΔNorth, ΔEast, ΔUp) kümülatif olarak integrate edilip
GPS referans noktasından Enlem/Boylam/İrtifa'ya dönüştürülür.
"""
import math, importlib.util, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Yollar ──────────────────────────────────────────────────────────────────
BASE    = Path(r"c:\Users\Lenovo\Desktop\projeler egit\SSBHava Araçlarında GPS Kullanmadan Konum Tahmini")
PH1     = BASE / "Phase1_Veri_Analizi" / "outputs"
PH2     = BASE / "Phase2_Model" / "outputs"
SCRIPTS = BASE / "Phase2_Model" / "scripts"
RAW_DIR = BASE / "Holybro Pixhawk Dataset" / "Holybro Pixhawk" / "processed" / "Grouped flights"
OUT_DIR = BASE / "Phase4_Rapor" / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model yükle ─────────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location("models", SCRIPTS / "models.py")
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
detrend_windows = _mod.detrend_windows
GRUModel        = _mod.GRUModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru = GRUModel().to(device)
gru.load_state_dict(torch.load(PH2 / "best_gru.pt", weights_only=True, map_location=device))
gru.eval()

R_EARTH = 6_371_000.0  # metre

SEQ_LEN = 40
STEP    = 4

def ned_to_latlon(lat0_deg, lon0_deg, alt0_m, cum_north_m, cum_east_m, cum_up_m):
    lat0 = math.radians(lat0_deg)
    lat  = lat0_deg + (cum_north_m / R_EARTH) * (180.0 / math.pi)
    lon  = lon0_deg + (cum_east_m  / (R_EARTH * math.cos(lat0))) * (180.0 / math.pi)
    alt  = alt0_m  + cum_up_m
    return lat, lon, alt

def make_windows(X_raw):
    N, F = X_raw.shape
    windows = []
    for i in range(0, N - SEQ_LEN + 1, STEP):
        windows.append(X_raw[i:i+SEQ_LEN])
    return np.stack(windows).astype(np.float32)  # (W, SEQ_LEN, F)

def get_ref_coords(flight_name):
    csv_path = RAW_DIR / f"{flight_name}_sync.csv"
    df = pd.read_csv(csv_path, nrows=10)
    ref_lat = df["ref_lat_f58"].dropna().iloc[0]
    ref_lon = df["ref_lon_f58"].dropna().iloc[0]
    ref_alt = df["ref_alt_f58"].dropna().iloc[0]
    return float(ref_lat), float(ref_lon), float(ref_alt)

def run_inference(X_raw):
    windows = make_windows(X_raw)
    wins_dt = detrend_windows(windows)
    t = torch.from_numpy(wins_dt).to(device)
    with torch.no_grad():
        preds = gru(t).cpu().numpy()  # (W, 3)
    return preds

def reconstruct_latlon(preds_delta, ref_lat, ref_lon, ref_alt, y_abs):
    """GRU tahminleri ve GPS gerçeği için Lat/Lon/Alt serileri üret."""
    W = len(preds_delta)

    # GRU kümülatif NED
    gru_cum_north = np.cumsum(preds_delta[:, 0])
    gru_cum_east  = np.cumsum(preds_delta[:, 1])
    gru_cum_up    = np.cumsum(preds_delta[:, 2])

    gru_lats, gru_lons, gru_alts = [], [], []
    for i in range(W):
        la, lo, al = ned_to_latlon(ref_lat, ref_lon, ref_alt,
                                    gru_cum_north[i], gru_cum_east[i], gru_cum_up[i])
        gru_lats.append(la); gru_lons.append(lo); gru_alts.append(al)

    # GPS gerçeği — y_abs doğrudan NED (metre) cinsinden, pencere adımlarıyla al
    step_idxs = [i * STEP + SEQ_LEN - 1 for i in range(W)]
    step_idxs = [min(s, len(y_abs) - 1) for s in step_idxs]
    true_north = y_abs[step_idxs, 0]
    true_east  = y_abs[step_idxs, 1]
    true_up    = -y_abs[step_idxs, 2]   # z_f58 = Down → Up = -z

    true_lats, true_lons, true_alts = [], [], []
    for i in range(W):
        la, lo, al = ned_to_latlon(ref_lat, ref_lon, ref_alt,
                                    true_north[i], true_east[i], true_up[i])
        true_lats.append(la); true_lons.append(lo); true_alts.append(al)

    return (np.array(gru_lats),  np.array(gru_lons),  np.array(gru_alts),
            np.array(true_lats), np.array(true_lons), np.array(true_alts))

# ── Demo: 3 uçuş ────────────────────────────────────────────────────────────
FLIGHTS = ["vuelo_100", "vuelo_105", "vuelo_110"]

fig = plt.figure(figsize=(18, 14))
fig.suptitle("GRU Model — Enlem/Boylam/İrtifa Tahmini vs GPS Gerçeği", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

results = []

for row, flight in enumerate(FLIGHTS):
    npz_path = PH1 / f"flight_{flight}_test.npz"
    d = np.load(npz_path)
    X_raw = d["X_scaled"]   # (N, 12)
    y_abs  = d["y_abs"]     # (N, 3)  — cumulative NED, metres

    ref_lat, ref_lon, ref_alt = get_ref_coords(flight)
    preds_delta = run_inference(X_raw)                # (W, 3)  — ΔN, ΔE, ΔU / pencere

    (gru_lats, gru_lons, gru_alts,
     true_lats, true_lons, true_alts) = reconstruct_latlon(
         preds_delta, ref_lat, ref_lon, ref_alt, y_abs)

    W = len(gru_lats)
    t = np.arange(W) * (STEP * 0.5)  # saniye

    # Son nokta hataları (derece → metre yaklaşımı)
    final_dlat_m = (gru_lats[-1] - true_lats[-1]) * (math.pi / 180) * R_EARTH
    final_dlon_m = (gru_lons[-1] - true_lons[-1]) * (math.pi / 180) * R_EARTH * math.cos(math.radians(ref_lat))
    final_hpe_m  = math.sqrt(final_dlat_m**2 + final_dlon_m**2)
    final_alt_err = abs(gru_alts[-1] - true_alts[-1])

    results.append({
        "flight": flight,
        "ref_lat_deg":  float(ref_lat),
        "ref_lon_deg":  float(ref_lon),
        "ref_alt_m":    float(ref_alt),
        "n_steps":      int(W),
        "flight_dur_s": float(t[-1]),
        "final_HPE_m":  round(float(final_hpe_m), 2),
        "final_alt_err_m": round(float(final_alt_err), 2),
        "gru_final_lat":  round(float(gru_lats[-1]), 7),
        "gru_final_lon":  round(float(gru_lons[-1]), 7),
        "gru_final_alt_m": round(float(gru_alts[-1]), 2),
        "true_final_lat": round(float(true_lats[-1]), 7),
        "true_final_lon": round(float(true_lons[-1]), 7),
        "true_final_alt_m": round(float(true_alts[-1]), 2),
    })

    # ── Grafik 1: Enlem-Boylam 2D rotası
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.plot(true_lons,  true_lats,  'b-',  lw=1.5, label="GPS Gerçeği", alpha=0.8)
    ax1.plot(gru_lons,   gru_lats,   'r--', lw=1.5, label="GRU Tahmin",  alpha=0.8)
    ax1.plot(true_lons[0],  true_lats[0],  'go', ms=6)
    ax1.plot(true_lons[-1], true_lats[-1], 'bs', ms=6)
    ax1.plot(gru_lons[-1],  gru_lats[-1],  'r^', ms=6)
    ax1.set_xlabel("Boylam (°)", fontsize=8)
    ax1.set_ylabel("Enlem (°)",  fontsize=8)
    ax1.set_title(f"{flight}\nRota (HPE_son={final_hpe_m:.1f}m)", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)
    ax1.ticklabel_format(useOffset=False, style='plain')

    # ── Grafik 2: İrtifa vs Zaman
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.plot(t, true_alts, 'b-',  lw=1.5, label="GPS İrtifa")
    ax2.plot(t, gru_alts,  'r--', lw=1.5, label="GRU İrtifa")
    ax2.set_xlabel("Zaman (s)", fontsize=8)
    ax2.set_ylabel("İrtifa (m)",fontsize=8)
    ax2.set_title(f"{flight}\nİrtifa (Δalt_son={final_alt_err:.1f}m)", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)

    # ── Grafik 3: Enlem/Boylam hataları metre cinsinden
    dlat_m = (gru_lats - true_lats) * (math.pi / 180) * R_EARTH
    dlon_m = (gru_lons - true_lons) * (math.pi / 180) * R_EARTH * math.cos(math.radians(ref_lat))
    hpe    = np.sqrt(dlat_m**2 + dlon_m**2)
    ax3 = fig.add_subplot(gs[row, 2])
    ax3.plot(t, hpe, 'purple', lw=1.5)
    ax3.fill_between(t, 0, hpe, alpha=0.15, color='purple')
    ax3.set_xlabel("Zaman (s)", fontsize=8)
    ax3.set_ylabel("HPE (m)", fontsize=8)
    ax3.set_title(f"{flight}\nKümülatif HPE", fontsize=9)
    ax3.tick_params(labelsize=7)

out_png = OUT_DIR / "ned_to_latlon_demo.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Grafik kaydedildi: {out_png}")

# ── JSON sonuçları ───────────────────────────────────────────────────────────
out_json = BASE / "Phase4_Rapor" / "outputs" / "ned_latlon_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Sonuçlar kaydedildi: {out_json}")

print("\n=== NED → Enlem/Boylam/İrtifa Özet ===")
print(f"{'Uçuş':<15} {'Süre(s)':>8} {'HPE_son(m)':>12} {'ΔAlt(m)':>10}")
print("-" * 50)
for r in results:
    print(f"{r['flight']:<15} {r['flight_dur_s']:>8.0f} {r['final_HPE_m']:>12.1f} {r['final_alt_err_m']:>10.1f}")

print("\nÖrnek dönüşüm (vuelo_100, son adım):")
r0 = results[0]
print(f"  Referans : {r0['ref_lat_deg']:.6f}°N, {r0['ref_lon_deg']:.6f}°E, {r0['ref_alt_m']:.1f}m")
print(f"  GPS son  : {r0['true_final_lat']:.6f}°N, {r0['true_final_lon']:.6f}°E, {r0['true_final_alt_m']:.1f}m")
print(f"  GRU son  : {r0['gru_final_lat']:.6f}°N, {r0['gru_final_lon']:.6f}°E, {r0['gru_final_alt_m']:.1f}m")
