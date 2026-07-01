# GPS Kullanmadan UAV Konum Tahmini — Sunum Scripti
## SSB TEKNOFEST 2026 | 10–15 Dakika

> Bu dosya slayt slayt konuşma notları + ekran içeriklerini içerir.
> Her slaytta **[EKRAN]** bölümü → slayta yazılacak metin/grafik
> **[KONUŞMA]** bölümü → sunucunun söyleyeceği (yaklaşık süre)

---

## SLAYT 1 — Kapak (30 sn)

**[EKRAN]**
```
GPS Kullanmadan UAV Konum Tahmini
Deep Learning ile IMU/Manyetometre/Baro Füzyonu

SSB TEKNOFEST 2026
[Ad Soyad]
19 Temmuz 2026
```

**[KONUŞMA]**
"Merhaba, bugün GPS sinyali kesildiğinde ya da karıştırıldığında
bir İHA'nın konumunu nasıl tahmin edebileceğimizi göstereceğim.
Yalnızca IMU, manyetometre, barometre ve hava hızı sensörlerini kullanarak
metre düzeyinde hassasiyet elde ettik."

---

## SLAYT 2 — Problem ve Motivasyon (60 sn)

**[EKRAN]**
```
Neden GPS-Free Navigasyon?

❌  Jamming (elektronik karıştırma)
❌  Spoofing (sinyal aldatma)
❌  Uydu görünürlüğü kaybı
❌  IMU tek başına → hızla sürüklenir (drift)

✅  Çözüm: IMU + Mag + Baro → Derin Öğrenme
```
*[Görsel: GPS-jammed bölge haritası veya sinyal bozulma şeması]*

**[KONUŞMA]**
"GPS navigasyonu birçok kritik senaryoda güvenilir değil.
Jamming elektronik harp ortamlarında kaçınılmaz, spoofing ise tespit edilmesi zor.
Geleneksel çözüm olan ataletsel navigasyon tek başına saniyeler içinde metreler sürüklüyor.
Biz bu sürüklenmeyi makine öğrenmesiyle baskılamayı hedefledik."

---

## SLAYT 3 — Veri Seti (60 sn)

**[EKRAN]**
```
Holybro Pixhawk — PX4 Autopilot uLog Kayıtları

┌─────────────────────────────────────┐
│  120 uçuş → 118 geçerli            │
│  ~2 Hz örnekleme                   │
│  250–360 saniye / uçuş             │
│  ~3003 sütun (tüm topic'ler)       │
└─────────────────────────────────────┘

GİRDİ (12 özellik):
  IMU (6) · Manyetometre (3) · Baro (1) · Airspeed (2)

HEDEF:
  ΔNorth, ΔEast, ΔUp  [m/adım — GPS asla girdi değil]
```

**[KONUŞMA]**
"Gerçek PX4 uçuş kayıtlarını kullandık — 118 uçuş.
Kritik nokta: GPS sütunları modele girdi olarak kesinlikle verilmedi.
x_f58, y_f58, z_f58 yalnızca hedef etiket olarak kullanıldı.
12 özellik: 6 IMU, 3 manyetometre, 1 baro ve 2 hava hızı."

---

## SLAYT 4 — Veri Ön İşleme (75 sn)

**[EKRAN]**
```
Pipeline:

1. Kalite filtresi:  < 100 satır → hariç (2 uçuş)
2. f-İndeks eşleme: PX4 topic kayması → adaptif çözüm
3. StandardScaler:  sadece train setinden fit
4. detrend_windows(): ← ÖZGÜN KATKI

   Test set Mag X std = Train std × 13×
   Çözüm: Her pencere içinde per-window z-score
   → Mag + Baro için uçuşlar arası tutarsızlık giderildi

5. Veri bölme: UÇUŞ bazlı (satır bazlı shuffle → veri sızıntısı!)
   Train: 80 uçuş | Val: 16 | Test: 22
```
*[Grafik: mag_baro_anomaly.png]*

**[KONUŞMA]**
"En önemli teknik katkımız detrend_windows fonksiyonu.
Test setindeki manyetometre standardı, eğitim setinin 13 katıydı.
Global normalizasyon yetersiz kalıyordu.
Her 40 adımlık pencere içinde z-score uygulayarak bu tutarsızlığı çözdük.
Bu fonksiyon olmadan model test setinde çok daha kötü performans gösteriyordu."

---

## SLAYT 5 — Model Mimarileri (75 sn)

**[EKRAN]**
```
4 Model Karşılaştırması:

GRU (Ana Model)          LSTM (Karşılaştırma)
──────────────────       ──────────────────────
hidden=128, 2 katman     hidden=128, 2 katman
153,987 parametre        205,187 parametre
12.9s eğitim             27.5s eğitim

AttentionGRU             Dilated CNN
──────────────────       ──────────────────────
GRU + softmax attn       3× Conv1D (dilation 1,2,4)
154,116 parametre        76,739 parametre (en küçük)

Hiperparametreler:  HuberLoss · Adam lr=1e-3
                    Pencere: 40 adım = 20 saniye
                    Batch: 256 · Erken durdurma: 15 epoch
```

**[KONUŞMA]**
"4 farklı mimari test ettik.
GRU ve LSTM temel rekabetçi modeller.
AttentionGRU, GRU üzerine ağırlıklı zamansal dikkat ekliyor.
Dilated CNN ise sıralı bellek olmadan uzun menzilli bağlam yakalamaya çalışıyor.
Hepsini aynı hiperparametrelerle eğittik, adil karşılaştırma için."

---

## SLAYT 6 — Eğitim Sonuçları (45 sn)

**[EKRAN]**
| Model | Val Loss | Best Epoch | Params |
|-------|----------|-----------|--------|
| **GRU** | **0.1789** | 70 | 154K |
| LSTM | 0.1852 | 63 | 205K |
| AttentionGRU | 0.2106 | 73 | 154K |
| CNN | 0.6112 | 83 | 77K |

*[Grafik: gru_training_loss.png — train vs val eğrisi]*

**[KONUŞMA]**
"GRU en iyi validation loss'u elde etti: 0.179.
LSTM çok yakın: 0.185 — neredeyse eşdeğer.
AttentionGRU GRU'dan %18 daha kötü — kısa pencerelerde GRU'nun gating'i
zaten örtük dikkat görevi görüyor, ayrı katman fayda sağlamıyor.
CNN ise alıcı alanı yetersiz kaldığından belirgin şekilde geride."

---

## SLAYT 7 — Test Performansı (75 sn)

**[EKRAN]**
```
Per-Adım HPE — Test Seti (2470 pencere, 22 uçuş)

                GRU        LSTM    AttGRU    CNN      Dead Reck.
HPE Ort.(m)   4.12  ✅    4.39    5.23     5.38      8.55
HPE Med.(m)   1.55  ✅    1.68    2.16     3.91      8.42
HPE P95 (m)  15.3         15.9   15.8    13.8  ✅   10.6
RMSE 3D (m)   3.80  ✅    3.98    4.44     3.97

GRU → Dead Reckoning'i %52 geride bıraktı (4.12m vs 8.55m)
CNN → en iyi p95: uç hataları kırpmada avantajlı
```
*[Grafik: hpe_comparison_boxplot.png]*

**[KONUŞMA]**
"GRU ana metrikte kazanıyor: ortalama 4.12 metre yatay hata.
Dead Reckoning'in iki katı daha iyi.
İlginç bulgu: CNN ortalamada kötü ama p95 en düşük — uç hatalar az.
Güvenlik kritik sistemlerde uç hataları önlemek önemli, bu durumda CNN tercih edilebilir.
İrtifa tahmini ise çok başarılı: RMSE sadece 0.37 m — baro sensörü çok güçlü."

---

## SLAYT 8 — GPS Kesintisi Deneyi (90 sn)

**[EKRAN]**
```
Protokol: 22 uçuş × 3 başlangıç × 4 süre = 264 senaryo

                    10s      30s      60s    Kalan Tümü
GRU (%20 başl.)   51m ✅    95m ✅   166m      456m
LSTM              72m      122m     263m      640m
Dead Reckoning   152m      438m     217m      400m

GRU vs Dead Reckoning:
  10s → %66 iyileşme   30s → %78 iyileşme   60s → %24 iyileşme
```
*[Grafik: outage_heatmap.png veya hpe_vs_time_f20_30s.png]*

**[KONUŞMA]**
"Bu deney projenin kalbidir.
GPS'i uçuşun %20'sinde kestik ve modelin ne kadar süre konum tutabileceğini ölçtük.
10 saniyede GRU 51 metre hata — Dead Reckoning 152 metre.
30 saniyede GRU 95 metre, Dead Reckoning 438 metre — %78 iyileşme!
60 saniye sonra avantaj azalıyor, kümülatif hata birikiyor.
Gerçek sistemde 10-30 saniyelik GPS kesintisi sorunsuz atlatılabilir."

---

## SLAYT 9 — Sensör Ablasyon Analizi (60 sn)

**[EKRAN]**
```
Hangi sensör ne kadar önemli?
(Feature zeroing — test setinde 2470 pencere)

Ablasyon            HPE Ort.   Artış
────────────────────────────────────────
Baseline (tam)       4.12 m     —
Mag yok (cols 6-8)  5.41 m    +31.2% ⚠️
Baro yok (col 10)   5.21 m    +26.3% ⚠️
Hava hızı yok       4.33 m    +4.9%
Yalnızca IMU        7.59 m    +84.1% 🔴
```
*[Grafik: sensor_ablation_hpe.png]*

**[KONUŞMA]**
"Her sensörü sıfırlayarak önem sıralaması çıkardık.
Manyetometre kaldırıldığında hata %31 artıyor — yön bilgisi kritik.
Baro olmadan %26 artış — irtifa tutarlılığı yatay tahmini de etkiliyor.
Yalnızca IMU ile model Dead Reckoning'e geri dönüyor: 7.59 metre.
Bu bulgu, uçuşlar arası mag tutarsızlığımızı nicel olarak doğruluyor."

---

## SLAYT 10 — Uçuş Koşulu Analizi (60 sn)

**[EKRAN]**
```
Hangi koşullar hatayı artırıyor?
(22 test uçuşu — Pearson korelasyon)

Özellik              Pearson r   Güç
─────────────────────────────────────
Mag Std (gürültü)    +0.547 🔴   GÜÇLÜ
Gyro Std (manevra)   -0.312      ORTA
Hava hızı            -0.187      zayıf

En iyi uçuş:  vuelo_116  → HPE = 0.81 m
En kötü uçuş: vuelo_117  → HPE = 7.32 m
```
*[Grafik: flight_error_correlation.png + per_flight_hpe_ranking.png]*

**[KONUŞMA]**
"Model hangi koşullarda zorlanıyor?
Manyetometre gürültüsü tek başına HPE'nin %55'ini açıklıyor.
Bu ablasyon bulgularımızla mükemmel örtüşüyor.
Manevralı uçuşlarda ise model daha iyi — gyro aktivitesi r=-0.31 ile negatif korelasyon.
Bu sezgisel: manevra sırasında ivme bilgisi daha belirgin, model daha kolay öğreniyor."

---

## SLAYT 11 — Gerçek Zamanlı Demo (45 sn)

**[EKRAN]**
```
realtime_inference.py — Satır Satır Akış Simülasyonu

Sensör akışı → Sliding Window (40 adım)
           → detrend_single()
           → GRU Model
           → ΔNorth, ΔEast, ΔUp
           → NED → Enlem/Boylam/İrtifa (çıktı)

python realtime_inference.py --flight vuelo_100

Per-adım gecikme: < 1ms (CPU)  |  Gerçek zamanlı uygulanabilir
```
*[Grafik: realtime_demo_vuelo_100.png]*

**[KONUŞMA]**
"Modeli gerçek zamanlı kullanım senaryosunda test ettik.
Her yeni sensör satırı geldiğinde pencere kaydırılıyor, model tahmin yapıyor.
CPU'da bile milisaniyenin altında gecikme — gömülü sistemlere aktarılabilir.
Bu bonus çalışmamızdır ve gerçek dünya uygulanabilirliğini gösteriyor."

---

## SLAYT 12 — NED → Enlem/Boylam Dönüşümü (30 sn)

**[EKRAN]**
```python
# Model çıktısı: ΔNorth, ΔEast, ΔUp (m/adım)
# GPS kesintisi başlangıcındaki son konum: lat0, lon0, alt0

lat = lat0 + (cum_north / R_earth) × (180/π)
lon = lon0 + (cum_east  / (R_earth × cos(lat0_rad))) × (180/π)
alt = alt0 + cum_up

Referans: 39.2991°N, -0.6150°E  (Holybro test sahası)
```
*[Grafik: ned_to_latlon_demo.png]*

**[KONUŞMA]**
"Model çıktısını GPS koordinatına dönüştürmek için haversine ters formülü kullandık.
Temel bölme değişkeni cos(lat0) — yüksek enlemlerde boylam derecesi daha küçük mesafe.
Bu dönüşüm Bölüm 3'teki problem tanımını doğrudan karşılıyor."

---

## SLAYT 13 — Karşılaşılan Sorunlar ve Çözümler (45 sn)

**[EKRAN]**
```
Problem → Çözüm

1. f-İndeks Kayması (PX4)
   Farklı uçuşlarda farklı topic instance numaraları
   → Adaptif find_findex() fonksiyonu yazıldı
   → 118/120 uçuş kurtarıldı

2. Manyetometre Std Tutarsızlığı (13×)
   → detrend_windows() ile per-pencere z-score

3. dur_steps STEP Karışıklığı (GPS kesinti deneyi)
   STEP=4 (pencere) ≠ FS=2 (örnekleme)
   → dur_steps = dur_s × FS olarak düzeltildi

4. LSTM Ağırlık Üzerine Yazılması
   → Aynı hiperparametrelerle yeniden eğitim (Δ0.126m)
```

**[KONUŞMA]**
"Her projede beklenmedik sorunlar çıkıyor.
En kritiki f-indeks kayması — PX4 formatına özgü bir karmaşıklık.
detrend_windows ise bir bug'ı çözmek için değil,
veri dağılımındaki gerçek bir sorunu gidermek için geliştirdiğimiz orijinal katkı."

---

## SLAYT 14 — Sonuçlar ve Gelecek (60 sn)

**[EKRAN]**
```
Elde Edilen Sonuçlar:
  ✅ GRU: per-adım HPE = 4.12 m (Dead Reckoning %52 daha iyi)
  ✅ GPS 10s kesintisi → 51 m (%66 iyileşme)
  ✅ GPS 30s kesintisi → 95 m (%78 iyileşme)
  ✅ İrtifa RMSE = 0.37 m (çok kararlı)
  ✅ Gerçek zamanlı inference < 1ms

Gelecek Çalışmalar:
  → Transformer/Seq2Seq dikkat mekanizması
  → MC Dropout ile belirsizlik tahmini
  → ONNX → Pixhawk gömülü sistem deployment
  → Tightly-coupled GPS/INS hibrit entegrasyon
```

**[KONUŞMA]**
"Özet olarak: GPS olmadan 4 metre HPE elde ettik.
10-30 saniyelik GPS kesintilerini güvenle atlattık.
Manyetometrenin kritik rolünü hem ablasyon hem korelasyon analizleriyle kanıtladık.
Gelecekte gömülü sisteme aktarma ve hibrit navigasyon entegrasyonu ana hedef.
Dinlediğiniz için teşekkürler."

---

## SLAYT 15 — Sorular (opsiyonel yedek slaylar)

**[EKRAN]**
```
Teşekkürler — Sorular

İletişim: [email]
Kod: GitHub / teslim paketi
```

---

## Yedek Slaytlar (Soru olursa göster)

### Y1 — Neden BiLSTM kullanmadın?
```
Gerçek zamanlı inference → gelecekteki ölçümlere erişim yok
BiLSTM yalnızca offline analizde kullanılabilir
Bu proje realtime uygulamayı hedeflediğinden causal (tek yönlü) model seçildi
```

### Y2 — CNN neden kötü?
```
Dilated CNN etkin alıcı alanı: ~15 / 40 timestep
GRU bellek mekanizması → tüm 40 adım üzerinden kümülatif bağlam
Navigasyon görevi uzun vadeli bağlam gerektiriyor → RNN üstünlüğü beklenen
```

### Y3 — 4.12 metre yeterli mi?
```
Karşılaştırma:
  • Dead Reckoning (fiziksel): 8.55m
  • Bu çalışma (GRU): 4.12m
  • GPS tipik (iyi koşullar): 1-3m
10-30s kesinti için: 51-95m final HPE (kümülatif)
Operasyonel bağlama göre: kısa kesintiler (< 30s) için kabul edilebilir
```

### Y4 — Veri seti yeterince çeşitli mi?
```
118 uçuş aynı test sahasından (İspanya, 39.3°N)
Manyetik alan sabiti → farklı coğrafyada genelleme riski
SpeedyBee dış doğrulama ile genellenebilirlik artırılabilir
```

---

## Süre Planı (toplam ~13 dakika)

| Slayt | İçerik | Süre |
|-------|--------|------|
| 1 | Kapak | 30s |
| 2 | Motivasyon | 60s |
| 3 | Veri seti | 60s |
| 4 | Ön işleme | 75s |
| 5 | Mimariler | 75s |
| 6 | Eğitim | 45s |
| 7 | Test performansı | 75s |
| 8 | GPS kesintisi | 90s |
| 9 | Ablasyon | 60s |
| 10 | Uçuş koşulu | 60s |
| 11 | Gerçek zamanlı demo | 45s |
| 12 | NED dönüşüm | 30s |
| 13 | Sorunlar/çözümler | 45s |
| 14 | Sonuçlar | 60s |
| 15 | Sorular | — |
| **TOPLAM** | | **~13 dk** |
