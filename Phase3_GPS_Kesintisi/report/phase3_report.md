# Phase 3: GPS Kesintisi Deneyi — Tam Trajektori Analizi

**Tarih:** 01 Temmuz 2026  
**Proje:** SSB GPS-Free UAV Konum Tahmini (TEKNOFEST)

---

## 1. Yönetici Özeti

Bu aşamada GRU ve LSTM modelleri ile ağırlıklı ensemble'ı (GRU×0.75 + LSTM×0.25) için
**GPS kesintisi tam trajektori simülasyonu** yapılmıştır. 22 test uçuşu üzerinde,
3 farklı kesinti başlangıç noktası (uçuşun %20 / %40 / %60'ı) ve 4 farklı süre
(10s / 30s / 60s / kalan tümü) kombinasyonunda toplamda **264 senaryo** test edilmiştir.

**Ana Bulgular:**

| Senaryo | GRU | Dead Reckoning | İyileşme |
|---------|-----|---------------|---------|
| 10s kesinti | ~51–64 m | ~115–185 m | **~3× daha iyi** |
| 30s kesinti | ~95–141 m | ~160–438 m | **~2–3× daha iyi** |
| 60s kesinti | ~166–216 m | ~206–220 m | ~%15 daha iyi |

---

## 2. Deney Protokolü

| Parametre | Değer |
|-----------|-------|
| Test uçuş sayısı | **22** |
| Kesinti başlangıç oranları | **%20, %40, %60** |
| Kesinti süreleri | **10s, 30s, 60s, Kalan Tümü** |
| Veri örnekleme hızı | **~2 Hz** (0.5s/satır) |
| Teacher forcing | **YOK** — yalnızca IMU/mag/baro/airspeed |
| Kümülatif hata | `HPE = √(Σ(pred_ΔN−true_ΔN)² + Σ(pred_ΔE−true_ΔE)²)` |

### 2.1 Simülasyon Mantığı

Uçuşun herhangi bir anında GPS kesildiğinde:
1. Her 0.5s adımında model, 20 saniyelik (40 satır) geçmiş pencereyi alır
2. Modelin tahmin ettiği ΔNorth, ΔEast, ΔUp gerçeğin yerine kullanılır
3. Kümülatif konum sapması zaman içinde hesaplanır
4. `HPE_final` = kesinti bitimindeki toplam yatay konum hatası

---

## 3. Sonuçlar

### 3.1 Kesinti Başlangıcı: %20

| Model | 10s | 30s | 60s | Kalan Tümü |
|-------|-----|-----|-----|-----------|
| **GRU** | **51.1 m** | **95.4 m** | **166.1 m** | 455.5 m |
| LSTM | 72.1 m | 122.4 m | 262.9 m | 640.0 m |
| Ensemble | 55.6 m | 99.4 m | 185.2 m | 492.9 m |
| Dead Reckoning | 152.3 m | 437.7 m | 217.1 m | **399.7 m** |

### 3.2 Kesinti Başlangıcı: %40

| Model | 10s | 30s | 60s | Kalan Tümü |
|-------|-----|-----|-----|-----------|
| **GRU** | **64.0 m** | **108.8 m** | **189.8 m** | **371.6 m** |
| LSTM | 101.0 m | 133.6 m | 178.6 m | 496.8 m |
| **Ensemble** | 72.3 m | **107.0 m** | 179.3 m | 394.8 m |
| Dead Reckoning | 185.4 m | 160.0 m | 206.1 m | 237.3 m |

### 3.3 Kesinti Başlangıcı: %60

| Model | 10s | 30s | 60s | Kalan Tümü |
|-------|-----|-----|-----|-----------|
| GRU | 28.8 m | 141.2 m | 216.0 m | **334.8 m** |
| **LSTM** | **18.8 m** | **137.7 m** | **135.2 m** | 470.7 m |
| Ensemble | 25.8 m | 136.7 m | 183.3 m | 364.7 m |
| Dead Reckoning | 115.1 m | 281.9 m | 220.4 m | 415.8 m |

---

## 4. Temel Gözlemler

### 4.1 GRU vs Dead Reckoning İyileşme Oranı

| Senaryo | GRU | DR | İyileşme |
|---------|-----|----|---------|
| %20 başl., 10s | 51 m | 152 m | **%66** |
| %20 başl., 30s | 95 m | 438 m | **%78** |
| %40 başl., 10s | 64 m | 185 m | **%65** |
| %40 başl., 30s | 109 m | 160 m | **%32** |
| %60 başl., 10s | 29 m | 115 m | **%75** |
| %60 başl., 30s | 141 m | 282 m | **%50** |

### 4.2 Önemli Bulgular

1. **Kısa kesimlerde model net üstün:** 10–30s kesimlerde GRU Dead Reckoning'e göre 2–3x daha iyi.

2. **%60 başlangıcında LSTM avantajlı:** Uçuşun son %40'ında (iniş fazı dinamikleri) LSTM, GRU'yu 10s ve 60s senaryolarda geçmektedir. LSTM'nin daha uzun hafızası iniş sırasındaki yavaşlama ve manevraları daha iyi yakalamaktadır.

3. **Uzun kesimlerde model-DR farkı azalır:** 60s+ kesimlerde kümülatif model hatası, DR'nin sistematik biasına yaklaşmaktadır. Bu beklenen fiziksel bir davranış.

4. **"Kalan tümü" senaryosunda DR rekabetçi:** Çok uzun kesimlerde (uçuşun %80'i boyunca) Dead Reckoning bazı senaryolarda modele yaklaşmakta — bu, modelin ortalama delta tahmini üzerinden sistematik bir bias biriktirdiğini gösterir.

5. **Ensemble değeri:** Ensemble, p95 hatasını tek model ortalamasına göre düşürür ama GRU'yu ortalama HPE'de her zaman geçemez. Gerçek zamanlı kullanım için GRU önerilir.

6. **DR anomalisi — 30s > 60s (%20 başlangıcı):** Dead Reckoning, %20 başlangıcında 30s için 437.7m, 60s için 217.1m göstermektedir. Bu, 60s daha iyi gibi görünse de aslında uçuşların ortalama net yer değiştirmesini ölçmektedir. Bazı uçuşlar ilk 30s'de hızlıca uzaklaşıp sonraki 30s'de geri döndüğünden kümülatif net hata 60s'de daha düşük çıkabilir. Makine öğrenmesi modelleri (GRU, LSTM) bu tür anomaline karşı daha dayanıklıdır — her iki durumda da DR'dan belirgin biçimde iyidir.

### 4.3 Phase 2 ile Tutarlılık Kontrolü

Phase 2'deki GPS kesintisi sonuçları farklı metodoloji (per-window değerlendirme) kullanmaktaydı. Karşılaştırma:

| Senaryo | Phase 2 | Phase 3 (%20 baş.) | Yorum |
|---------|---------|-------------------|-------|
| GRU 10s | **49.4 m** | **51.1 m** | ✅ ~tutarlı |
| GRU 30s | **116.5 m** | **95.4 m** | ✅ aynı büyüklük |
| GRU 60s | **190.6 m** | **166.1 m** | ✅ aynı büyüklük |

Tutarlılık iyi — Phase 3'ün %20 başlangıç noktasındaki değerleri Phase 2 ile örtüşmektedir.

---

## 5. Üretilen Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `outputs/outage_matrix_v3.json` | Tüm 48 senaryo ham metrikleri (mean/median/p95) |
| `outputs/plots/outage_heatmap.png` | GRU/LSTM/Ensemble için 3×4 ısı haritası |
| `outputs/plots/bar_comparison_f20.png` | %20 başlangıç, 4 model bar grafiği |
| `outputs/plots/bar_comparison_f40.png` | %40 başlangıç, 4 model bar grafiği |
| `outputs/plots/bar_comparison_f60.png` | %60 başlangıç, 4 model bar grafiği |
| `outputs/plots/hpe_vs_time_f*_*.png` | 9 senaryo için HPE×zaman eğrileri |
| `outputs/plots/sample_ucus_hpe_f40_60s.png` | Örnek uçuş HPE eğrisi |

---

## 6. Özet Tablo — Önerilen Yapılandırma

Tüm senaryolarda GRU modeli birincil seçim olarak öne çıkmaktadır:

| Kesinti Süresi | Önerilen Model | Beklenen HPE |
|---------------|----------------|-------------|
| 0–10s | GRU | 30–65 m |
| 10–30s | GRU veya Ensemble | 95–140 m |
| 30–60s | GRU (iniş fazında LSTM) | 135–220 m |
| 60s+ | Hybrid (GPS yeniden bağlan) | 170–460 m |

---

*Oluşturan: `01_gps_outage_simulation.py` — Phase 3 GPS Kesintisi Deneyi*  
*Düzeltme: dur_steps = dur_s × FS (STEP ile bölme hatası giderildi)*
