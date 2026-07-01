# Slayt Yapısı ve Grafik Referansları
## PowerPoint/Keynote Hazırlama Kılavuzu

---

## SLAYT LİSTESİ (15 slayt + 4 yedek)

| # | Başlık | Grafik Dosyası | Süre |
|---|--------|----------------|------|
| 1 | Kapak | — | 30s |
| 2 | Problem ve Motivasyon | *(jammer görseli — web'den)* | 60s |
| 3 | Veri Seti | `Phase1_Veri_Analizi/outputs/plots/data_quality_overview.png` | 60s |
| 4 | Veri Ön İşleme | `Phase1_Veri_Analizi/outputs/plots/mag_baro_anomaly.png` | 75s |
| 5 | Model Mimarileri | *(tablo + mini mimari şeması)* | 75s |
| 6 | Eğitim Sonuçları | `Phase2_Model/outputs/plots/gru_training_loss.png` | 45s |
| 7 | Test Performansı | `Phase2_Model/outputs/plots/hpe_comparison_boxplot.png` | 75s |
| 8 | GPS Kesintisi | `Phase3_GPS_Kesintisi/outputs/plots/outage_heatmap.png` | 90s |
| 9 | Sensör Ablasyon | `Phase2_Model/outputs/plots/sensor_ablation_hpe.png` | 60s |
| 10 | Uçuş Koşulu Analizi | `Phase4_Rapor/outputs/plots/flight_error_correlation.png` | 60s |
| 11 | Gerçek Zamanlı Demo | `Phase4_Rapor/outputs/plots/realtime_demo_vuelo_100.png` | 45s |
| 12 | NED → LatLon | `Phase4_Rapor/outputs/plots/ned_to_latlon_demo.png` | 30s |
| 13 | Sorunlar ve Çözümler | *(metin — tablo)* | 45s |
| 14 | Sonuçlar | *(özet tablo)* | 60s |
| 15 | Sorular | — | — |

---

## TASARIM ÖNERİLERİ

**Renk paleti:**
- Ana: `#1E3A5F` (koyu mavi) — başlıklar
- Vurgu: `#00A8E8` (açık mavi) — tablolarda en iyi değer
- Başarı: `#4CAF50` (yeşil) — ✅ işaretleri
- Uyarı: `#FF5722` (kırmızı) — kritik bulgular
- Arka plan: `#FAFAFA` veya beyaz

**Font:**
- Başlık: Calibri Bold 28-32 pt
- İçerik: Calibri 18-20 pt
- Kod bloğu: Consolas 14 pt (gri kutu arka plan)

**Genel kurallar:**
- Her slayta tek ana mesaj
- En fazla 5-6 madde
- Grafikleri büyük tut (slaytın %60+)
- Animasyon: fade in yeterli, karmaşık geçiş dikkat dağıtır

---

## KRİTİK SLAYTAR (Jüri bunlara odaklanır)

### Slayt 4 — detrend_windows (ÖZGÜN KATKI)
Bu slayt projenin teknik özgünlüğünü gösteriyor.
Grafik `mag_baro_anomaly.png` mutlaka gösterilmeli.
"13× std oranı" rakamı söylenirken grafiğe işaret et.

### Slayt 7 — Test Performansı (TAHMİN BAŞARISI %10)
GRU'nun Dead Reckoning'den %52 daha iyi olduğunu vurgula.
Jüri bu tabloyu fotoğraflayabilir — net ve okunabilir olsun.

### Slayt 8 — GPS Kesintisi (DENEYSEL ÇALIŞMA %25)
En ağırlıklı kriter. 264 senaryo sayısı dikkat çeker.
Isı haritası grafiği varsa mutlaka göster.

### Slayt 9 — Ablasyon (MODEL TASARIMI %30)
Feature zeroing yöntemi model anlayışını gösteriyor.
"+31% HPE (mag yok)" rakamı ezberlenmeli.

---

## DEĞERLENDİRME KRİTERLERİ KAPSAMASI

| Kriter | Ağırlık | İlgili Slaytar |
|--------|---------|----------------|
| Veri analizi ve ön işleme | %15 | 3, 4 |
| Model tasarımı | %30 | 5, 9, 10 |
| Tahmin başarısı | %10 | 7 |
| Deneysel çalışma | %25 | 8, 9, 10, 11 |
| Rapor içeriği | %10 | *(sunumda değil — raporda)* |
| Sunum içeriği | %10 | Tümü |

---

## OLASI JÜRI SORULARI

**S: "GPS verisini hiç kullanmadığınızı nasıl kanıtlıyorsunuz?"**
C: "Girdi sütun listesi models.py'da sabit. GPS_COLS listesi ayrı tutuldu,
INPUT_COLS'a dahil edilmedi. Rapordaki Bölüm 2.3'te belgelenmiş."

**S: "4 metre HPE pratik uygulamalar için yeterli mi?"**
C: "Bağlama göre değişir. Kısa GPS kesintileri (10-30s) için evet.
Gerçek sistemde hibrit GPS/INS entegrasyonu öneririz — bu çalışma
GPS yokken 10-30s arası güvenli köprü sağlar."

**S: "Modeli farklı bir drone'da test ettin mi?"**
C: "SpeedyBee dataset dış doğrulama için hazır (farklı platform, BFC firmware).
Zaman kısıtı nedeniyle bu çalışmada tamamlanamadı — gelecek çalışma olarak planlandı."

**S: "Neden Transformer kullanmadın?"**
C: "40 adımlık kısa pencerede Transformer'ın avantajı sınırlı.
AttentionGRU deneyi bunu kanıtladı: açık dikkat mekanizması GRU'yu geçemedi.
Seq2Seq Transformer daha uzun bağlamda anlam taşır — future work."

**S: "Real-time latency nedir?"**
C: "CPU'da < 1ms per tahmin. GRU 154K parametre — Raspberry Pi 4'te bile çalışır.
ONNX export ile Pixhawk flight controller'a aktarılabilir."
