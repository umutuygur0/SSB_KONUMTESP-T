# Phase 1 — Veri Analizi ve Ön İşleme Raporu

**Üretim tarihi:** 2026-06-30 (kurtarma sonrası güncellendi)

---

## 1. Veri Seti Özeti

| Parametre | Değer |
|---|---|
| Toplam CSV dosyası | 120 |
| Geçerli uçuş | 118 (111 normal + 7 kurtarılan) |
| Kurtarılamaz | 2 (vuelo_16: 14 satır, vuelo_113: 3 satır) |
| Toplam sütun (Holybro) | 3003 |
| Benzersiz f-indeks grubu | 141 |
| GPS leakage riski taşıyan sütun | 177 |

## 2. Reddedilen Uçuşlar

| file               |   row_count |   duration_s | reject_reason                                                                                |
|:-------------------|------------:|-------------:|:---------------------------------------------------------------------------------------------|
| vuelo_113_sync.csv |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'mag_field[0]_f49']    |
| vuelo_117_sync.csv |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'mag_field[0]_f49']    |
| vuelo_118_sync.csv |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'mag_field[0]_f49']    |
| vuelo_119_sync.csv |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'mag_field[0]_f49']    |
| vuelo_120_sync.csv |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'mag_field[0]_f49']    |
| vuelo_16_sync.csv  |          14 |            0 | Az satir: 14 < 100                                                                           |
| vuelo_7_sync.csv   |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'baro_alt_meter_f117'] |
| vuelo_8_sync.csv   |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'baro_alt_meter_f117'] |
| vuelo_9_sync.csv   |           0 |            0 | Eksik kritik sutun: ['delta_angle[0]_f124', 'delta_velocity[0]_f124', 'baro_alt_meter_f117'] |

## 3. Model Girdi Sütunları (GPS Leakage Yok)

Bulunan ve kullanılacak sütunlar (14):

```
delta_angle[0]_f124
delta_angle[1]_f124
delta_angle[2]_f124
delta_angle_dt_f124
delta_velocity[0]_f124
delta_velocity[1]_f124
delta_velocity[2]_f124
delta_velocity_dt_f124
mag_field[0]_f49
mag_field[1]_f49
mag_field[2]_f49
indicated_airspeed_m_s_f5
baro_alt_meter_f117
differential_pressure_pa_f15
```

## 4. Ön İşleme Parametreleri

| Parametre | Değer |
|---|---|
| Örnekleme hızı | ~2 Hz (senkronize) |
| Pencere uzunluğu | 40 adım (20 saniye) |
| Adım boyutu | 4 adım (2 saniye) |
| Feature sayısı | 12 |
| İşlenen uçuş | 118 |

## 5. Eğitim / Doğrulama / Test Bölümü

| Küme | Örnek Sayısı | Oran |
|---|---|---|
| Train | 9,677 | %70.6 |
| Val | 1,843 | %13.4 |
| Test | 2,470 | %18.0 |

Uçuş bazlı split: Train=80 | Val=16 | Test=22

## 6. Uçuş Süresi İstatistikleri

| Stat | Değer |
|---|---|
| Ortalama | 258.6 s |
| Medyan | 254.0 s |
| Min | 230.7 s |
| Maks | 357.2 s |
| Toplam | 7.97 saat |

## 7. Sensör İstatistikleri (3 Örnek Uçuştan)

| Sütun | Ortalama | Std | Min | Maks |
|---|---|---|---|---|
| `delta_angle[0]_f124` | -0.0000 | 0.0015 | -0.0070 | 0.0061 |
| `delta_angle[1]_f124` | 0.0003 | 0.0007 | -0.0015 | 0.0036 |
| `delta_angle[2]_f124` | 0.0001 | 0.0012 | -0.0035 | 0.0038 |
| `delta_velocity[0]_f124` | 0.0012 | 0.0050 | -0.0135 | 0.0178 |
| `delta_velocity[1]_f124` | -0.0003 | 0.0048 | -0.0164 | 0.0149 |
| `delta_velocity[2]_f124` | -0.0517 | 0.0128 | -0.1159 | 0.0000 |
| `mag_field[0]_f49` | -0.0005 | 0.0140 | -0.0536 | 0.0558 |
| `mag_field[1]_f49` | -0.0004 | 0.0052 | -0.0183 | 0.0204 |
| `mag_field[2]_f49` | 0.0006 | 0.0210 | -0.0673 | 0.0903 |
| `indicated_airspeed_m_s_f5` | 23.8726 | 2.5346 | 17.0538 | 31.0760 |
| `baro_alt_meter_f117` | -111.0024 | 25.6710 | -164.9970 | -59.5050 |
| `differential_pressure_pa_f15` | 360.5583 | 72.4812 | 153.3987 | 584.7376 |
| `x_f58` | 226.9126 | 135.3349 | 10.9535 | 447.9891 |
| `y_f58` | 61.2632 | 275.2355 | -565.9117 | 494.1347 |
| `z_f58` | -78.1577 | 21.1371 | -122.8342 | -38.1015 |

## 8. Üretilen Grafikler

- `outputs/plots/01_sensor_distributions.png`
- `outputs/plots/02_flight_tracks.png`
- `outputs/plots/03_altitude_profiles.png`
- `outputs/plots/04_quality_overview.png`
- `outputs/plots/05_nan_heatmap.png`
- `outputs/plots/06_imu_timeseries.png`

## 9. Phase 2 İçin Hazır Dosyalar

Toplam çıktı dosyası: 118
```
X_train.npy : shape=(9677, 40, 12)   — 80 uçuş
X_val.npy   : shape=(1843, 40, 12)   — 16 uçuş
X_test.npy  : shape=(2470, 40, 12)   — 22 uçuş
y_train.npy : shape=(9677, 3)
y_val.npy   : shape=(1843, 3)
y_test.npy  : shape=(2470, 3)
scaler.pkl  : StandardScaler, 80 uçuş ham veriden fit (mean≈0, std≈1)
118 × flight_*.npz : ham + ölçekli veri, per-uçuş
```