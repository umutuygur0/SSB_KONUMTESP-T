# GPS Kullanmadan UAV Konum Tahmini

**SSB TEKNOFEST** — GPS-Free UAV Position Estimation with Deep Learning

---

## Proje Hakkında

Bu proje, bir İHA'nın (UAV) GPS sinyali olmaksızın yalnızca
**IMU · Manyetometre · Barometre · Hava Hızı** sensörlerini kullanarak
anlık konumunu tahmin etmektedir.

Model çıktısı: **ΔNorth, ΔEast, ΔUp** (NED yer değiştirmesi, metre/adım)  
Ground truth: PX4 EKF lokal konum (`x_f58`, `y_f58`, `z_f58`)

---

## Sonuçlar

### Per-Adım Doğruluk (Test Seti — 2470 pencere, 22 uçuş)

| Model | HPE Ort. | HPE Medyan | HPE P95 | RMSE 3D |
|-------|----------|-----------|--------|--------|
| **GRU** | **4.12 m** | **1.55 m** | 15.3 m | **3.80 m** |
| LSTM | 4.39 m | 1.68 m | 15.9 m | 3.98 m |
| AttentionGRU | 5.23 m | 2.16 m | 15.8 m | 4.44 m |
| CNN (Dilated) | 5.38 m | 3.91 m | **13.8 m** | 3.97 m |
| Dead Reckoning | 8.55 m | 8.42 m | 10.6 m | 5.02 m |

### GPS Kesintisi (Trajektori, %20 Başlangıç)

| Süre | GRU | Dead Reckoning | İyileşme |
|------|-----|---------------|---------|
| 10s | 51 m | 152 m | **%66** |
| 30s | 95 m | 438 m | **%78** |
| 60s | 166 m | 217 m | **%24** |

---

## Veri Seti

**Holybro Pixhawk — PX4 Autopilot uLog** kayıtları  
120 uçuş → 118 geçerli (vuelo_1–vuelo_120)  
~2 Hz örnekleme, 250–360 saniye/uçuş

Veri seti gizlilik nedeniyle repoya dahil edilmemiştir (`.gitignore`).

---

## Kurulum

```bash
pip install torch numpy scikit-learn matplotlib pandas
```

GPU (CUDA) önerilir; CPU'da da çalışır.

---

## Kullanım

### Phase 1 — Veri Ön İşleme ve EDA

```bash
cd Phase1_Veri_Analizi/scripts
python 01_data_preprocessing.py   # 118 uçuşu işle, NPZ kaydet
python 02_sliding_windows.py      # 40-adım pencereler oluştur
python 03_train_val_test_split.py # 80/16/22 uçuş bölmesi
python 04_eda_visualizations.py   # EDA: 4 grafik (dağılım, korelasyon, kalite, anomali)
```

### Phase 2 — Model Eğitimi ve Analiz

```bash
cd Phase2_Model/scripts
python 02_train_gru.py            # GRU eğitimi → best_gru.pt
python 03_train_lstm.py           # LSTM eğitimi → best_lstm.pt
python 04_evaluate.py             # Test seti değerlendirme
python 05_sensor_ablation.py      # Sensör ablasyon (feature zeroing)
python 06_train_attention_gru.py  # AttentionGRU → best_attn_gru.pt
python 07_train_cnn.py            # Dilated CNN → best_cnn.pt
```

### Phase 3 — GPS Kesintisi Simülasyonu

```bash
cd Phase3_GPS_Kesintisi/scripts
python 01_gps_outage_simulation.py  # 264 senaryo matrisi (3×4×22 uçuş)
```

### Phase 4 — Grafikler ve Rapor

```bash
cd Phase4_Rapor/scripts
python 01_trajectory_plots.py       # 2D rota, irtifa, drift analizi
python 02_outage_boxplot.py         # Boxplot ve kombine karşılaştırma
python 03_ned_to_latlon_demo.py     # NED→LatLon dönüşüm ve görselleştirme
python 04_flight_error_analysis.py  # Uçuş koşulu korelasyon analizi
```

### Gerçek Zamanlı Demo

```bash
python realtime_inference.py --flight vuelo_100   # Satır satır akış simülasyonu
```

---

## Proje Yapısı

```
SSBHava Araçlarında GPS Kullanmadan Konum Tahmini/
├── Phase1_Veri_Analizi/
│   ├── scripts/
│   │   ├── 01_data_preprocessing.py
│   │   ├── 02_sliding_windows.py
│   │   ├── 03_train_val_test_split.py
│   │   └── 04_eda_visualizations.py   # EDA: 4 grafik
│   └── outputs/
│       ├── X_train.npy / X_val.npy / X_test.npy
│       ├── y_train.npy / y_val.npy / y_test.npy
│       ├── scaler.pkl
│       ├── flight_*_{train,val,test}.npz  (118 uçuş)
│       └── plots/                         # 4 EDA grafiği
├── Phase2_Model/
│   ├── scripts/
│   │   ├── models.py                  # GRU, LSTM, AttGRU, CNN, detrend_windows
│   │   ├── 02_train_gru.py
│   │   ├── 03_train_lstm.py
│   │   ├── 04_evaluate.py
│   │   ├── 05_sensor_ablation.py      # Ablasyon analizi
│   │   ├── 06_train_attention_gru.py
│   │   └── 07_train_cnn.py
│   ├── outputs/
│   │   ├── best_gru.pt / best_lstm.pt / best_attn_gru.pt / best_cnn.pt
│   │   ├── metrics_{gru,lstm,attn_gru,cnn}.json
│   │   ├── sensor_ablation.json
│   │   └── plots/                     # eğitim eğrileri, ablasyon
│   └── report/
├── Phase3_GPS_Kesintisi/
│   ├── scripts/
│   ├── outputs/                       # outage_matrix_v3.json, 14 grafik
│   └── report/
├── Phase4_Rapor/
│   ├── scripts/
│   │   ├── 01_trajectory_plots.py
│   │   ├── 02_outage_boxplot.py
│   │   ├── 03_ned_to_latlon_demo.py
│   │   └── 04_flight_error_analysis.py
│   ├── outputs/
│   │   ├── ned_latlon_results.json
│   │   ├── flight_error_analysis.json
│   │   └── plots/                     # tüm Phase 4 grafikleri
│   └── report/
│       └── final_report.md
├── realtime_inference.py              # Gerçek zamanlı demo
├── requirements.txt
└── README.md
```

---

## Teknik Detaylar

| Parametre | Değer |
|-----------|-------|
| Girdi boyutu | (batch, 40, 12) |
| Çıktı boyutu | (batch, 3) — ΔN, ΔE, ΔUp |
| GRU hidden size | 128 |
| LSTM hidden size | 128 |
| Loss fonksiyonu | HuberLoss (δ=1.0) |
| Optimizer | Adam (lr=1e-3) |
| Eğitim süresi | ~13s (CUDA) |
| Pencere uzunluğu | 40 adım = 20 saniye |

---

## Lisans

Akademik ve araştırma amaçlı kullanım için açıktır.

---

*SSB TEKNOFEST — GPS-Free UAV Konum Tahmini*
