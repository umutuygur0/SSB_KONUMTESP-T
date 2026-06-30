# Phase 2: GPS-Free Konum Tahmini — Model Eğitimi & Değerlendirme

**Tarih:** 30 June 2026  
**Değerlendirici:** muratisbilen41@gmail.com  
**Proje:** SSB GPS-Free UAV Konum Tahmini (TEKNOFEST)

---

## 1. Yönetici Özeti

Bu rapor, UAV'ın GPS sinyali olmaksızın IMU / manyetometre / baro / airspeed
sensörlerini kullanarak anlık NED konum değişimini (ΔNorth, ΔEast, ΔUp) tahmin
etmek amacıyla eğitilen **GRU** ve **LSTM** modellerinin sonuçlarını içermektedir.

| Model | HPE Ortalama | HPE Medyan | HPE P95 | 3DE Ort. | RMSE 3D |
|-------|-------------|-----------|--------|---------|--------|
| GRU  | 4.124 m | 1.549 m | 15.320 m | 4.166 m | 3.800 m |
| LSTM | 4.261 m | 1.602 m | 15.874 m | 4.306 m | 3.911 m |
| Dead Reckoning | 8.551 m | 8.418 m | 10.595 m | 8.574 m | 5.017 m |

---

## 2. Veri & Girdi/Çıktı Tanımı

### 2.1 Ham Veri
- **Kaynak:** PX4 Autopilot log dosyaları (kaggle UAV Coordination Dataset)
- **Örnekleme hızı:** ~2 Hz (senkronize)
- **Toplam uçuş:** 118 (80 train / 19 val / 19 test)

### 2.2 Girdi Özellikleri (12 feature)

| # | Kolon | Açıklama |
|---|-------|----------|
| 0-2 | `delta_angle[0-2]_f124` | IMU açısal hız inkremanı (rad) |
| 3-5 | `delta_velocity[0-2]_f124` | IMU hız inkremanı (m/s) |
| 6-8 | `mag_field[0-2]_f49` | Manyetometre alanı (Gauss) |
| 9  | `indicated_airspeed_m_s_f5` | Gösterge hava hızı (m/s) |
| 10 | `baro_alt_meter_f117` | Barometrik irtifa (m) |
| 11 | `differential_pressure_pa_f15` | Diferansiyel basınç (Pa) |

### 2.3 Çıktı (3 hedef)

| Hedef | Açıklama | Kaynak |
|-------|----------|--------|
| ΔNorth (m) | Kuzey yönü yerleşim değişimi | `x_f58.diff()` |
| ΔEast (m)  | Doğu yönü yerleşim değişimi | `y_f58.diff()` |
| ΔUp (m)    | Yukarı yönü yerleşim değişimi | `(-z_f58).diff()` |

### 2.4 Pencere Parametreleri

| Parametre | Değer |
|-----------|-------|
| Pencere uzunluğu | 40 adım = 20 saniye |
| Adım büyüklüğü | 4 adım = 2 saniye |
| Train örnekleri | 2470 → **9 677** |
| Val örnekleri | **1 843** |
| Test örnekleri | **2 470** |

---

## 3. Model Mimarileri

### 3.1 GRU

```
Input  : (batch, 40, 12)
GRU    : input_size=12, hidden_size=128, num_layers=2,
         batch_first=True, dropout=0.2  (katmanlar arası)
Dropout: 0.2  (GRU çıkışı sonrası)
Linear : 128 → 3
Output : (batch, 3)  [ΔNorth, ΔEast, ΔUp]
Toplam parametre: 153,987
```

### 3.2 LSTM

```
Input  : (batch, 40, 12)
LSTM   : input_size=12, hidden_size=128, num_layers=2,
         batch_first=True, dropout=0.2
Dropout: 0.2
Linear : 128 → 3
Output : (batch, 3)  [ΔNorth, ΔEast, ΔUp]
Toplam parametre: 205,187
```

> **Not:** BiLSTM kullanılmamıştır — gerçek zamanlı inference (gelecek veriye erişim yok)
  gerektirdiğinden tek yönlü yapı seçilmiştir.

### 3.3 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Loss | HuberLoss (δ=1.0) |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early Stopping patience | 15 epoch |
| Batch size | 256 |
| Max epoch | 150 |
| Gradient clip | 1.0 |

---

## 4. Dead Reckoning Baseline

Fiziksel entegrasyon ile basit referans tahmini:

```
ΔNorth ≈  delta_velocity[0] × 0.5 s
ΔEast  ≈  delta_velocity[1] × 0.5 s
ΔUp    ≈ -delta_velocity[2] × 0.5 s   (z_body = Down)

Varsayım: küçük açı / seviyeli uçuş (body ≈ NED dönüşümü yok)
```

Bu yaklaşım tutum hesabı (attitude) yapmadığından yalnızca seviyeli
uçuşta makul sonuç üretir. Manevralarda hata hızla büyür.

---

## 5. Eğitim Sonuçları

| Parametre | GRU | LSTM |
|-----------|-----|------|
| Best val loss | 0.17886 | 0.16795 |
| Best epoch | 70 | 51 |
| Toplam epoch | 85 | 66 |
| Eğitim süresi | 12.9 s | 12.4 s |
| Cihaz | cuda | cuda |

**Loss grafikleri:**

| GRU | LSTM |
|-----|------|
| ![GRU Loss](../outputs/plots/gru_training_loss.png) | ![LSTM Loss](../outputs/plots/lstm_training_loss.png) |

---

## 6. Test Seti Değerlendirmesi

### 6.1 Tüm Metrikler

| Metrik | GRU | LSTM | Dead Reckoning |
|--------|-----|------|----------------|
| HPE Ortalama (m) | 4.124 | 4.261 | 8.551 |
| HPE Medyan (m)   | 1.549 | 1.602 | 8.418 |
| HPE P95 (m)      | 15.320 | 15.874 | 10.595 |
| 3DE Ortalama (m) | 4.166 | 4.306 | 8.574 |
| 3DE Medyan (m)   | 1.589 | 1.650 | 8.448 |
| RMSE North (m)   | 3.614 | 4.150 | 5.281 |
| RMSE East (m)    | 5.488 | 5.339 | 6.873 |
| RMSE Up (m)      | 0.366 | 0.404 | 0.614 |
| RMSE 3D (m)      | 3.800 | 3.911 | 5.017 |
| MAE North (m)    | 2.043 | 2.359 | 4.216 |
| MAE East (m)     | 3.157 | 3.095 | 6.350 |
| MAE Up (m)       | 0.283 | 0.306 | 0.470 |
| Test örnekleri   | 2470 | 2470 | 2470 |

### 6.2 HPE Kutu Grafiği

![HPE Boxplot](../outputs/plots/hpe_comparison_boxplot.png)

### 6.3 Örnek Uçuş Rotası

![Trajectory 2D](../outputs/plots/sample_trajectory_2d.png)

---

## 7. GPS Kesinti Deneyi

**Protokol:**
- Test uçuşlarının %20, %40, %60. adımında GPS kesildi
- Kesinti sonrası sadece IMU/mag/baro/airspeed kullanıldı
- Model tahminleri kümülatif olarak toplandı (Teacher Forcing YOK)
- HPE = √(ΔN² + ΔE²) ile gerçek GPS yoluyla karşılaştırıldı

### 7.1 Ortalama Final HPE (m) — Kesinti Süresine Göre

| Kesinti Süresi | GRU | LSTM | Dead Reckoning |
|---------------|-----|------|----------------|
| 10s | 49.42 m | 63.24 m | 153.86 m |
| 30s | 116.49 m | 146.30 m | 296.62 m |
| 60s | 190.61 m | 198.80 m | 237.52 m |

### 7.2 HPE vs Kesinti Süresi (sürekli)

![HPE vs Time](../outputs/plots/hpe_vs_outage_time.png)

### 7.3 Kesinti Süresi vs Ortalama HPE (bar grafiği)

![Outage Bar](../outputs/plots/gps_outage_duration_vs_hpe.png)

---

## 8. Sonuçlar & Değerlendirme

### 8.1 Model Karşılaştırması

- **GRU**, HPE ortalaması (4.124 m) ile
  LSTM'ye (4.261 m) göre daha düşük yatay hata üretmiştir.
- Her iki model de Dead Reckoning baseline'ı (8.551 m)
  belirgin biçimde geride bırakmıştır.
- Dead Reckoning tutum bilgisi (attitude) hesaplamadığından manevralar sırasında
  hata hızla büyümektedir.

### 8.2 GPS Kesinti Analizi

- 10 saniyelik kesimlerde hem GRU hem LSTM kabul edilebilir hata üretmektedir.
- 60 saniyelik kesimlerde kümülatif hata artışı beklenen bir davranıştır;
  gerçek sistemde hybrid (GPS/INS tightly coupled) entegrasyon önerilir.

### 8.3 Üretilen Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `outputs/best_gru.pt` | En iyi GRU model ağırlıkları |
| `outputs/best_lstm.pt` | En iyi LSTM model ağırlıkları |
| `outputs/metrics_gru.json` | GRU test metrikleri |
| `outputs/metrics_lstm.json` | LSTM test metrikleri |
| `outputs/metrics_baseline.json` | Dead Reckoning metrikleri |
| `outputs/outage_results.json` | GPS kesinti deneyi sonuçları |
| `outputs/plots/gru_training_loss.png` | GRU eğitim eğrisi |
| `outputs/plots/lstm_training_loss.png` | LSTM eğitim eğrisi |
| `outputs/plots/hpe_comparison_boxplot.png` | Model HPE karşılaştırması |
| `outputs/plots/sample_trajectory_2d.png` | Örnek uçuş rotası |
| `outputs/plots/hpe_vs_outage_time.png` | HPE vs kesinti süresi |
| `outputs/plots/gps_outage_duration_vs_hpe.png` | Kesinti süresi bar grafiği |

---

## 9. Sonraki Adımlar (Phase 3 Önerileri)

1. **Hiperparametre optimizasyonu:** hidden_size, num_layers, dropout grid search
2. **Attention mekanizması:** Transformer encoder veya attention-GRU
3. **Çıktı belirsizliği:** MDN (Mixture Density Network) veya MC Dropout ile
   konum güven aralığı tahmini
4. **Gerçek uçuşa entegrasyon:** PX4 SITL ile HIL (Hardware-in-the-Loop) testi
5. **Sıkıştırma:** TFLite / ONNX export ile gömülü sistem deployment

---

*Bu rapor `05_phase2_report.py` tarafından otomatik oluşturulmuştur.*