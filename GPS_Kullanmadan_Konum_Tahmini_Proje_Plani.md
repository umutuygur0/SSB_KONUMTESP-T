# GPS Kullanmadan Konum Tahmini — Proje Planı

> **Teslim:** 19 Temmuz 2026 Pazar 23:59 | **Değerlendirici:** muratisbilen41@gmail.com

---

## Net karar

**Ana veri seti:** `Holybro Pixhawk` — 120 uçuş, 1046 sütun, 9–12 MB/uçuş  
**Bonus / dış doğrulama:** `Speedybee DataSet` — 110+ Lap, GPS ve telemetri ayrı dosyalar  
**Kaggle UAV Coordination Dataset:** raporda alternatif olarak anılır, model eğitiminde kullanılmaz

---

## 1. Neden Kaggle uygun değil?

- Sentetik veri, görev-seviyesi tablosal yapı
- Ham yüksek frekanslı IMU verisi yok (timestamp 1-dakika aralıklı)
- Yalnızca 2 MB, ~2000 satır
- Projenin ihtiyacı: zamanla sıralanmış gerçek uçuş, ham ivmeölçer/jiroskop/manyetometre/barometre/hava hızı + GPS ground truth
- Raporda "incelendi, yetersiz bulundu" diye belirtilecek

---

## 2. Veri Seti Gerçek Yapısı

### Holybro Pixhawk (Ana)

| Parametre | Değer |
|---|---|
| Uçuş dosyası | 120 CSV (vuelo_1 – vuelo_120) |
| Kullanılabilir | **118** (vuelo_113 → 3 satır; vuelo_16 → 15 satır, hariç tutulacak) |
| Sütun sayısı | **1046** |
| GPS sütunları | `lat_f43`, `lon_f43`, `alt_f43` |
| NED local position | `x_f58`, `y_f58`, `z_f58` (EKF çıktısı, ground truth hedefi) |
| NED referansı | `ref_lat_f58`, `ref_lon_f58`, `ref_alt_f58` |
| Ham gyro | `delta_angle[0-2]_f124` |
| Ham ivme | `delta_velocity[0-2]_f124` |
| Mag | `mag_field[0-2]_f49` |
| Airspeed | `indicated_airspeed_m_s_f5` |
| Baro irtifa | `baro_alt_meter_f117` |
| Diff basınç | `differential_pressure_pa_f15` |

**f-indeks notu:** Sütunlar PX4 uLog topic instance numarasıyla etiketlenmiş. f124 = sensor_combined (ham IMU), f49-f57 = magnetometer örnekleri, f58-f60 = vehicle_local_position örnekleri. f61-f63 = vehicle_angular_velocity.

### SpeedyBee (Bonus / Dış Doğrulama)

GPS ve telemetri **ayrı dosyalar** — uçuş timestamp'i üzerinden birleştirme gerekli.

| Dosya | İçerik |
|---|---|
| `Lap00X_gps.csv` | `GPS_coord[0,1]`, `GPS_altitude`, `GPS_speed`, `GPS_home_lat/lon` |
| `Lap00X.csv` (telemetri) | `gyroADC[0-2]`, `accSmooth[0-2]`, `magADC[0-2]`, `BaroAlt` ✓ |

**SpeedyBee GPS Leakage Listesi (model girdisinden kesinlikle çıkarılacak):**
```
navPos[0], navPos[1], navPos[2]       ← INAV GPS-fused yerel konum
navTgtPos[0], navTgtPos[1], navTgtPos[2]  ← waypoint hedef konumu
fwPosP, fwPosI, fwPosD, fwPosOut     ← GPS-fused PID kontrol sinyalleri
GPS_coord[*], GPS_altitude            ← doğrudan GPS
GPS_home_lat, GPS_home_lon            ← GPS referans
```

---

## 3. GPS Leakage — Yasaklı Girdi Listesi (Holybro)

```
# Doğrudan GPS
lat_f43, lon_f43, alt_f43
lat_f44, lon_f44, alt_f44
lat_f45, lon_f45, alt_f45

# EKF local/global position (sadece TARGET olarak kullanılacak, asla INPUT değil)
x_f58, y_f58, z_f58, vx_f58, vy_f58, vz_f58
x_f59, y_f59, z_f59
x_f60, y_f60, z_f60

# GPS referans noktaları
ref_lat_f58, ref_lon_f58, ref_alt_f58
ref_lat_f59, ref_lon_f59, ref_alt_f59

# GPS'ten türetilmiş rüzgar tahmini (EKF çıktısı)
windspeed_north_f7, windspeed_east_f7
windspeed_north_f8, windspeed_east_f8

# GPS fix ve kalite metrikleri
lat_lon_valid_f43, lat_lon_reset_counter_f43
terrain_alt_f43
```

---

## 4. Kullanılacak Model Girdileri

```python
INPUT_COLS = [
    # Ham gyro (sensor_combined delta_angle)
    "delta_angle[0]_f124", "delta_angle[1]_f124", "delta_angle[2]_f124",
    "delta_angle_dt_f124",
    # Ham ivme (sensor_combined delta_velocity)
    "delta_velocity[0]_f124", "delta_velocity[1]_f124", "delta_velocity[2]_f124",
    "delta_velocity_dt_f124",
    # Manyetometre
    "mag_field[0]_f49", "mag_field[1]_f49", "mag_field[2]_f49",
    # Hava hızı
    "indicated_airspeed_m_s_f5",
    # Barometre
    "baro_alt_meter_f117",
    # Diferansiyel basınç (airspeed doğrulaması)
    "differential_pressure_pa_f15",
]

TARGET_COLS = [
    "x_f58",   # NED North (metre)
    "y_f58",   # NED East (metre)
    "z_f58",   # NED Down (metre, negatif = yukarı)
]

# Modelin ürettiği output: ΔNorth, ΔEast, ΔUp (her timestep için)
# z ekseni: -z_f58 = yukarı yön (Up)
```

---

## 5. Hedef Değişken Stratejisi

Görev "Enlem, Boylam, İrtifa" istiyor. Doğrudan tahmin etmek yerine NED yer değiştirmesi tahmin edilip dönüştürülecek:

```
P0 = GPS kesintisi anındaki son konum (ref_lat_f58, ref_lon_f58, ref_alt_f58)
Her adımda model: ΔNorth_t, ΔEast_t, ΔUp_t üretir
P_t = P_0 + Σ ΔP_i
Son adımda NED → Enlem/Boylam/İrtifa dönüşümü yapılır (haversine ters)
```

Neden doğrudan enlem-boylam değil:
- Değerler 10^1 büyüklüğünde, değişimler 10^-4 büyüklüğünde → yüksek R² ama büyük metrik hata
- NED metre cinsinden olduğu için kayıp fonksiyonu anlamlı ve yorumlanabilir

---

## 6. Model Yapısı

### Baseline: Inertial Dead Reckoning (fiziksel)
- İvmeyi yönelimle dünya koordinatlarına çevir → yer çekimini çıkar → entegre et
- Karşılaştırma için referans noktası

### Model 1: GRU (Causal)
```
Sampling: 20 Hz
Sequence length: 100 (5 saniye)
Input features: 14
GRU layers: 2 × hidden_size=128
Dropout: 0.2
Output: ΔNorth, ΔEast, ΔUp
Loss: Huber
Optimizer: Adam, lr=1e-3
```

### Model 2: LSTM (Causal)
Aynı mimari, karşılaştırma için.

**BiLSTM kullanılmıyor** — gelecekteki ölçümleri okur → GPS kesintisi senaryosuna uygulanamaz.

---

## 7. Veri Ön İşleme Pipeline

### Aşama 1 — Kalite Filtresi
```
Minimum satır sayısı: 500 (25 saniye @ 20Hz)
Minimum uçuş süresi: 30 saniye
IMU sütunları boş olmamalı
Baro sütunu boş olmamalı
Timestamp monoton artan olmalı
```
Bilinen kötü dosyalar: **vuelo_113** (3 satır), **vuelo_16** (15 satır)

### Aşama 2 — Sütun Seçimi
- Girdi: INPUT_COLS listesi
- Hedef: TARGET_COLS listesi
- GPS/GNSS sütunları hiçbir işlemde model girdisi olarak kullanılmaz

### Aşama 3 — Örnekleme Hizalama
```
Hedef frekans: 20 Hz
IMU (delta_angle/delta_velocity): downsample
Baro, hava hızı: linear interpolate
Magnetometre: forward-fill + interpolate
Boşluk limiti: 1 saniyeden uzun interpolasyon yapılmaz
```

### Aşama 4 — Veri Bölme (uçuş bazlı)
```
Eğitim : vuelo 1–83   (~%70)
Doğrulama: vuelo 84–100  (~%15)
Test    : vuelo 101–120  (~%15)
```
Satır bazlı shuffle kesinlikle yapılmaz — veri sızıntısı yaratır.

### Aşama 5 — Normalizasyon
```python
StandardScaler — per-feature, sadece eğitim setinden fit edilir
```
Scaler test/doğrulama setine transform olarak uygulanır, yeniden fit edilmez.

---

## 8. GPS Kesintisi Deneyi

Test uçuşlarında kesinti matrisi:

| Kesinti başlangıcı | Kesinti süresi |
|---|---|
| Uçuşun %20'si | 10s / 30s / 60s / kalan tümü |
| Uçuşun %40'ı | 10s / 30s / 60s / kalan tümü |
| Uçuşun %60'ı | 10s / 30s / 60s / kalan tümü |

**Önemli:** Test sırasında modele önceki **tahmin** verilir, gerçek konum değil. Teacher forcing gerçekçi olmayan sonuç verir.

---

## 9. Metrikler

```
HPE    = sqrt((ΔN)² + (ΔE)²)        — Yatay konum hatası (metre)
3D_E   = sqrt((ΔN)² + (ΔE)² + (ΔU)²)
Ort. HPE, RMSE, Medyan, %95, Endpoint error, Drift rate (m/s)
Kesinti sürelerine göre ayrı raporlama: 10s / 30s / 60s
```

Enlem-boylam hatasını derece değil **metre** cinsinden raporla.

---

## 10. Beklenen Grafikler

1. Train–validation loss grafiği
2. Gerçek ve tahmin edilen 2B rota (Enlem-Boylam)
3. Gerçek ve tahmin edilen 3B rota
4. Zamana karşı yatay konum hatası
5. Zamana karşı irtifa hatası
6. Kesinti süresi vs. ortalama hata
7. Dead reckoning / GRU / LSTM karşılaştırması
8. Farklı GPS kesilme noktaları kutu grafiği

---

## 11. Çalışma Takvimi

| Aşama | İçerik | Durum |
|---|---|---|
| Phase 1 | Veri analizi, ön işleme, kalite filtresi, split | 🔄 Devam ediyor |
| Phase 2 | Dead reckoning baseline + GRU/LSTM eğitimi | Bekliyor |
| Phase 3 | GPS kesintisi deneyleri | Bekliyor |
| Phase 4 | Metrikler, grafikler, teknik rapor, README | Bekliyor |
| Phase 5 | Sunum + kod paketi + test + teslim | Bekliyor |

---

## 12. Bonus

- GRU ve LSTM karşılaştırması (değerlendirme bonusu)
- SpeedyBee üzerinde dış doğrulama
- Gerçek zamanlı satır/sensör akışı simülasyonu

---

## Klasör Yapısı

```
SSBHava Araçlarında GPS Kullanmadan Konum Tahmini/
├── Phase1_Veri_Analizi/
│   ├── scripts/
│   ├── outputs/
│   └── report/
├── Phase2_Model/
├── Phase3_GPS_Kesintisi/
├── Phase4_Rapor/
├── Holybro Pixhawk Dataset/
├── Speedybee DataSet/
├── kaggle UAV Coordination Dataset/
├── GPS_Kullanmadan_Konum_Tahmini_Proje_Plani.md
└── generaltask.txt
```
