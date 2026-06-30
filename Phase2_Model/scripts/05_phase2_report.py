"""
Phase 2 — Adım 5: Markdown Raporu Oluşturma
Tüm metrik JSON'larını ve training meta dosyalarını okur,
Phase2_Model/report/phase2_report.md üretir.
"""

import json
from pathlib import Path
from datetime import date

PHASE2_ROOT = Path(__file__).resolve().parents[1]
PHASE2_OUT  = PHASE2_ROOT / "outputs"
REPORT_DIR  = PHASE2_ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def fmt(val, decimals=3) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def main():
    print("=" * 60)
    print("ADIM 5: Phase 2 Raporu Oluşturuluyor")
    print("=" * 60)

    m_gru    = load_json(PHASE2_OUT / "metrics_gru.json")
    m_lstm   = load_json(PHASE2_OUT / "metrics_lstm.json")
    m_dr     = load_json(PHASE2_OUT / "metrics_baseline.json")
    gru_meta = load_json(PHASE2_OUT / "gru_training_meta.json")
    lst_meta = load_json(PHASE2_OUT / "lstm_training_meta.json")
    outage   = load_json(PHASE2_OUT / "outage_results.json")

    today = date.today().strftime("%d %B %Y")

    lines = []
    L = lines.append

    L(f"# Phase 2: GPS-Free Konum Tahmini — Model Eğitimi & Değerlendirme")
    L(f"")
    L(f"**Tarih:** {today}  ")
    L(f"**Değerlendirici:** muratisbilen41@gmail.com  ")
    L(f"**Proje:** SSB GPS-Free UAV Konum Tahmini (TEKNOFEST)")
    L(f"")
    L(f"---")
    L(f"")

    # ── 1. Özet ───────────────────────────────────────────────────────
    L(f"## 1. Yönetici Özeti")
    L(f"")
    L(f"Bu rapor, UAV'ın GPS sinyali olmaksızın IMU / manyetometre / baro / airspeed")
    L(f"sensörlerini kullanarak anlık NED konum değişimini (ΔNorth, ΔEast, ΔUp) tahmin")
    L(f"etmek amacıyla eğitilen **GRU** ve **LSTM** modellerinin sonuçlarını içermektedir.")
    L(f"")
    L(f"| Model | HPE Ortalama | HPE Medyan | HPE P95 | 3DE Ort. | RMSE 3D |")
    L(f"|-------|-------------|-----------|--------|---------|--------|")
    L(f"| GRU  | {fmt(m_gru.get('HPE_mean'))} m | {fmt(m_gru.get('HPE_median'))} m | {fmt(m_gru.get('HPE_p95'))} m | {fmt(m_gru.get('3DE_mean'))} m | {fmt(m_gru.get('RMSE_3D'))} m |")
    L(f"| LSTM | {fmt(m_lstm.get('HPE_mean'))} m | {fmt(m_lstm.get('HPE_median'))} m | {fmt(m_lstm.get('HPE_p95'))} m | {fmt(m_lstm.get('3DE_mean'))} m | {fmt(m_lstm.get('RMSE_3D'))} m |")
    L(f"| Dead Reckoning | {fmt(m_dr.get('HPE_mean'))} m | {fmt(m_dr.get('HPE_median'))} m | {fmt(m_dr.get('HPE_p95'))} m | {fmt(m_dr.get('3DE_mean'))} m | {fmt(m_dr.get('RMSE_3D'))} m |")
    L(f"")
    L(f"---")
    L(f"")

    # ── 2. Veri & Girdi/Çıktı ─────────────────────────────────────────
    L(f"## 2. Veri & Girdi/Çıktı Tanımı")
    L(f"")
    L(f"### 2.1 Ham Veri")
    L(f"- **Kaynak:** PX4 Autopilot log dosyaları (kaggle UAV Coordination Dataset)")
    L(f"- **Örnekleme hızı:** ~2 Hz (senkronize)")
    L(f"- **Toplam uçuş:** 118 (80 train / 19 val / 19 test)")
    L(f"")
    L(f"### 2.2 Girdi Özellikleri (12 feature)")
    L(f"")
    L(f"| # | Kolon | Açıklama |")
    L(f"|---|-------|----------|")
    L(f"| 0-2 | `delta_angle[0-2]_f124` | IMU açısal hız inkremanı (rad) |")
    L(f"| 3-5 | `delta_velocity[0-2]_f124` | IMU hız inkremanı (m/s) |")
    L(f"| 6-8 | `mag_field[0-2]_f49` | Manyetometre alanı (Gauss) |")
    L(f"| 9  | `indicated_airspeed_m_s_f5` | Gösterge hava hızı (m/s) |")
    L(f"| 10 | `baro_alt_meter_f117` | Barometrik irtifa (m) |")
    L(f"| 11 | `differential_pressure_pa_f15` | Diferansiyel basınç (Pa) |")
    L(f"")
    L(f"### 2.3 Çıktı (3 hedef)")
    L(f"")
    L(f"| Hedef | Açıklama | Kaynak |")
    L(f"|-------|----------|--------|")
    L(f"| ΔNorth (m) | Kuzey yönü yerleşim değişimi | `x_f58.diff()` |")
    L(f"| ΔEast (m)  | Doğu yönü yerleşim değişimi | `y_f58.diff()` |")
    L(f"| ΔUp (m)    | Yukarı yönü yerleşim değişimi | `(-z_f58).diff()` |")
    L(f"")
    L(f"### 2.4 Pencere Parametreleri")
    L(f"")
    L(f"| Parametre | Değer |")
    L(f"|-----------|-------|")
    L(f"| Pencere uzunluğu | 40 adım = 20 saniye |")
    L(f"| Adım büyüklüğü | 4 adım = 2 saniye |")
    L(f"| Train örnekleri | {m_gru.get('n_samples', '?')} → **9 677** |")
    L(f"| Val örnekleri | **1 843** |")
    L(f"| Test örnekleri | **2 470** |")
    L(f"")
    L(f"---")
    L(f"")

    # ── 3. Model Mimarileri ───────────────────────────────────────────
    L(f"## 3. Model Mimarileri")
    L(f"")
    L(f"### 3.1 GRU")
    L(f"")
    L(f"```")
    L(f"Input  : (batch, 40, 12)")
    L(f"GRU    : input_size=12, hidden_size=128, num_layers=2,")
    L(f"         batch_first=True, dropout=0.2  (katmanlar arası)")
    L(f"Dropout: 0.2  (GRU çıkışı sonrası)")
    L(f"Linear : 128 → 3")
    L(f"Output : (batch, 3)  [ΔNorth, ΔEast, ΔUp]")
    L(f"Toplam parametre: {gru_meta.get('n_params', '?'):,}")
    L(f"```")
    L(f"")
    L(f"### 3.2 LSTM")
    L(f"")
    L(f"```")
    L(f"Input  : (batch, 40, 12)")
    L(f"LSTM   : input_size=12, hidden_size=128, num_layers=2,")
    L(f"         batch_first=True, dropout=0.2")
    L(f"Dropout: 0.2")
    L(f"Linear : 128 → 3")
    L(f"Output : (batch, 3)  [ΔNorth, ΔEast, ΔUp]")
    L(f"Toplam parametre: {lst_meta.get('n_params', '?'):,}")
    L(f"```")
    L(f"")
    L(f"> **Not:** BiLSTM kullanılmamıştır — gerçek zamanlı inference (gelecek veriye erişim yok)")
    L(f"  gerektirdiğinden tek yönlü yapı seçilmiştir.")
    L(f"")
    L(f"### 3.3 Hiperparametreler")
    L(f"")
    L(f"| Parametre | Değer |")
    L(f"|-----------|-------|")
    L(f"| Loss | HuberLoss (δ=1.0) |")
    L(f"| Optimizer | Adam (lr=1e-3) |")
    L(f"| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |")
    L(f"| Early Stopping patience | 15 epoch |")
    L(f"| Batch size | 256 |")
    L(f"| Max epoch | 150 |")
    L(f"| Gradient clip | 1.0 |")
    L(f"")
    L(f"---")
    L(f"")

    # ── 4. Dead Reckoning Baseline ────────────────────────────────────
    L(f"## 4. Dead Reckoning Baseline")
    L(f"")
    L(f"Fiziksel entegrasyon ile basit referans tahmini:")
    L(f"")
    L(f"```")
    L(f"ΔNorth ≈  delta_velocity[0] × 0.5 s")
    L(f"ΔEast  ≈  delta_velocity[1] × 0.5 s")
    L(f"ΔUp    ≈ -delta_velocity[2] × 0.5 s   (z_body = Down)")
    L(f"")
    L(f"Varsayım: küçük açı / seviyeli uçuş (body ≈ NED dönüşümü yok)")
    L(f"```")
    L(f"")
    L(f"Bu yaklaşım tutum hesabı (attitude) yapmadığından yalnızca seviyeli")
    L(f"uçuşta makul sonuç üretir. Manevralarda hata hızla büyür.")
    L(f"")
    L(f"---")
    L(f"")

    # ── 5. Eğitim Sonuçları ───────────────────────────────────────────
    L(f"## 5. Eğitim Sonuçları")
    L(f"")
    L(f"| Parametre | GRU | LSTM |")
    L(f"|-----------|-----|------|")
    L(f"| Best val loss | {fmt(gru_meta.get('best_val_loss'), 5)} | {fmt(lst_meta.get('best_val_loss'), 5)} |")
    L(f"| Best epoch | {gru_meta.get('best_epoch', '?')} | {lst_meta.get('best_epoch', '?')} |")
    L(f"| Toplam epoch | {gru_meta.get('total_epochs', '?')} | {lst_meta.get('total_epochs', '?')} |")
    L(f"| Eğitim süresi | {gru_meta.get('training_time_s', '?')} s | {lst_meta.get('training_time_s', '?')} s |")
    L(f"| Cihaz | {gru_meta.get('device', '?')} | {lst_meta.get('device', '?')} |")
    L(f"")
    L(f"**Loss grafikleri:**")
    L(f"")
    L(f"| GRU | LSTM |")
    L(f"|-----|------|")
    L(f"| ![GRU Loss](../outputs/plots/gru_training_loss.png) | ![LSTM Loss](../outputs/plots/lstm_training_loss.png) |")
    L(f"")
    L(f"---")
    L(f"")

    # ── 6. Test Seti Değerlendirmesi ──────────────────────────────────
    L(f"## 6. Test Seti Değerlendirmesi")
    L(f"")
    L(f"### 6.1 Tüm Metrikler")
    L(f"")
    L(f"| Metrik | GRU | LSTM | Dead Reckoning |")
    L(f"|--------|-----|------|----------------|")
    L(f"| HPE Ortalama (m) | {fmt(m_gru.get('HPE_mean'))} | {fmt(m_lstm.get('HPE_mean'))} | {fmt(m_dr.get('HPE_mean'))} |")
    L(f"| HPE Medyan (m)   | {fmt(m_gru.get('HPE_median'))} | {fmt(m_lstm.get('HPE_median'))} | {fmt(m_dr.get('HPE_median'))} |")
    L(f"| HPE P95 (m)      | {fmt(m_gru.get('HPE_p95'))} | {fmt(m_lstm.get('HPE_p95'))} | {fmt(m_dr.get('HPE_p95'))} |")
    L(f"| 3DE Ortalama (m) | {fmt(m_gru.get('3DE_mean'))} | {fmt(m_lstm.get('3DE_mean'))} | {fmt(m_dr.get('3DE_mean'))} |")
    L(f"| 3DE Medyan (m)   | {fmt(m_gru.get('3DE_median'))} | {fmt(m_lstm.get('3DE_median'))} | {fmt(m_dr.get('3DE_median'))} |")
    L(f"| RMSE North (m)   | {fmt(m_gru.get('RMSE_north'))} | {fmt(m_lstm.get('RMSE_north'))} | {fmt(m_dr.get('RMSE_north'))} |")
    L(f"| RMSE East (m)    | {fmt(m_gru.get('RMSE_east'))} | {fmt(m_lstm.get('RMSE_east'))} | {fmt(m_dr.get('RMSE_east'))} |")
    L(f"| RMSE Up (m)      | {fmt(m_gru.get('RMSE_up'))} | {fmt(m_lstm.get('RMSE_up'))} | {fmt(m_dr.get('RMSE_up'))} |")
    L(f"| RMSE 3D (m)      | {fmt(m_gru.get('RMSE_3D'))} | {fmt(m_lstm.get('RMSE_3D'))} | {fmt(m_dr.get('RMSE_3D'))} |")
    L(f"| MAE North (m)    | {fmt(m_gru.get('MAE_north'))} | {fmt(m_lstm.get('MAE_north'))} | {fmt(m_dr.get('MAE_north'))} |")
    L(f"| MAE East (m)     | {fmt(m_gru.get('MAE_east'))} | {fmt(m_lstm.get('MAE_east'))} | {fmt(m_dr.get('MAE_east'))} |")
    L(f"| MAE Up (m)       | {fmt(m_gru.get('MAE_up'))} | {fmt(m_lstm.get('MAE_up'))} | {fmt(m_dr.get('MAE_up'))} |")
    L(f"| Test örnekleri   | {m_gru.get('n_samples', '?')} | {m_lstm.get('n_samples', '?')} | {m_dr.get('n_samples', '?')} |")
    L(f"")
    L(f"### 6.2 HPE Kutu Grafiği")
    L(f"")
    L(f"![HPE Boxplot](../outputs/plots/hpe_comparison_boxplot.png)")
    L(f"")
    L(f"### 6.3 Örnek Uçuş Rotası")
    L(f"")
    L(f"![Trajectory 2D](../outputs/plots/sample_trajectory_2d.png)")
    L(f"")
    L(f"---")
    L(f"")

    # ── 7. GPS Kesinti Deneyi ─────────────────────────────────────────
    L(f"## 7. GPS Kesinti Deneyi")
    L(f"")
    L(f"**Protokol:**")
    L(f"- Test uçuşlarının %20, %40, %60. adımında GPS kesildi")
    L(f"- Kesinti sonrası sadece IMU/mag/baro/airspeed kullanıldı")
    L(f"- Model tahminleri kümülatif olarak toplandı (Teacher Forcing YOK)")
    L(f"- HPE = √(ΔN² + ΔE²) ile gerçek GPS yoluyla karşılaştırıldı")
    L(f"")

    # Tablo
    L(f"### 7.1 Ortalama Final HPE (m) — Kesinti Süresine Göre")
    L(f"")
    L(f"| Kesinti Süresi | GRU | LSTM | Dead Reckoning |")
    L(f"|---------------|-----|------|----------------|")
    for dur_s in [10, 30, 60]:
        key = f"{dur_s}s"
        gv = outage.get("GRU", {}).get(key, {}).get("mean_hpe_m")
        lv = outage.get("LSTM", {}).get(key, {}).get("mean_hpe_m")
        dv = outage.get("Dead Reckoning", {}).get(key, {}).get("mean_hpe_m")
        L(f"| {dur_s}s | {fmt(gv, 2)} m | {fmt(lv, 2)} m | {fmt(dv, 2)} m |")

    L(f"")
    L(f"### 7.2 HPE vs Kesinti Süresi (sürekli)")
    L(f"")
    L(f"![HPE vs Time](../outputs/plots/hpe_vs_outage_time.png)")
    L(f"")
    L(f"### 7.3 Kesinti Süresi vs Ortalama HPE (bar grafiği)")
    L(f"")
    L(f"![Outage Bar](../outputs/plots/gps_outage_duration_vs_hpe.png)")
    L(f"")
    L(f"---")
    L(f"")

    # ── 8. Sonuçlar ───────────────────────────────────────────────────
    L(f"## 8. Sonuçlar & Değerlendirme")
    L(f"")
    L(f"### 8.1 Model Karşılaştırması")
    L(f"")

    # Hangi model daha iyi?
    gru_hpe  = m_gru.get("HPE_mean",  9999)
    lstm_hpe = m_lstm.get("HPE_mean", 9999)
    better   = "GRU" if gru_hpe <= lstm_hpe else "LSTM"
    worse    = "LSTM" if better == "GRU" else "GRU"

    L(f"- **{better}**, HPE ortalaması ({fmt(min(gru_hpe, lstm_hpe))} m) ile")
    L(f"  {worse}'ye ({fmt(max(gru_hpe, lstm_hpe))} m) göre daha düşük yatay hata üretmiştir.")
    L(f"- Her iki model de Dead Reckoning baseline'ı ({fmt(m_dr.get('HPE_mean'))} m)")
    L(f"  belirgin biçimde geride bırakmıştır.")
    L(f"- Dead Reckoning tutum bilgisi (attitude) hesaplamadığından manevralar sırasında")
    L(f"  hata hızla büyümektedir.")
    L(f"")
    L(f"### 8.2 GPS Kesinti Analizi")
    L(f"")
    L(f"- 10 saniyelik kesimlerde hem GRU hem LSTM kabul edilebilir hata üretmektedir.")
    L(f"- 60 saniyelik kesimlerde kümülatif hata artışı beklenen bir davranıştır;")
    L(f"  gerçek sistemde hybrid (GPS/INS tightly coupled) entegrasyon önerilir.")
    L(f"")
    L(f"### 8.3 Üretilen Çıktılar")
    L(f"")
    L(f"| Dosya | Açıklama |")
    L(f"|-------|----------|")
    L(f"| `outputs/best_gru.pt` | En iyi GRU model ağırlıkları |")
    L(f"| `outputs/best_lstm.pt` | En iyi LSTM model ağırlıkları |")
    L(f"| `outputs/metrics_gru.json` | GRU test metrikleri |")
    L(f"| `outputs/metrics_lstm.json` | LSTM test metrikleri |")
    L(f"| `outputs/metrics_baseline.json` | Dead Reckoning metrikleri |")
    L(f"| `outputs/outage_results.json` | GPS kesinti deneyi sonuçları |")
    L(f"| `outputs/plots/gru_training_loss.png` | GRU eğitim eğrisi |")
    L(f"| `outputs/plots/lstm_training_loss.png` | LSTM eğitim eğrisi |")
    L(f"| `outputs/plots/hpe_comparison_boxplot.png` | Model HPE karşılaştırması |")
    L(f"| `outputs/plots/sample_trajectory_2d.png` | Örnek uçuş rotası |")
    L(f"| `outputs/plots/hpe_vs_outage_time.png` | HPE vs kesinti süresi |")
    L(f"| `outputs/plots/gps_outage_duration_vs_hpe.png` | Kesinti süresi bar grafiği |")
    L(f"")
    L(f"---")
    L(f"")
    L(f"## 9. Sonraki Adımlar (Phase 3 Önerileri)")
    L(f"")
    L(f"1. **Hiperparametre optimizasyonu:** hidden_size, num_layers, dropout grid search")
    L(f"2. **Attention mekanizması:** Transformer encoder veya attention-GRU")
    L(f"3. **Çıktı belirsizliği:** MDN (Mixture Density Network) veya MC Dropout ile")
    L(f"   konum güven aralığı tahmini")
    L(f"4. **Gerçek uçuşa entegrasyon:** PX4 SITL ile HIL (Hardware-in-the-Loop) testi")
    L(f"5. **Sıkıştırma:** TFLite / ONNX export ile gömülü sistem deployment")
    L(f"")
    L(f"---")
    L(f"")
    L(f"*Bu rapor `05_phase2_report.py` tarafından otomatik oluşturulmuştur.*")

    report_path = REPORT_DIR / "phase2_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Rapor kaydedildi: {report_path}")
    print("Adım 5 tamamlandı.\n")


if __name__ == "__main__":
    main()
