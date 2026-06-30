"""
Phase 1 — Adım 5: Otomatik Rapor Üretici
Tüm adımlardan elde edilen JSON/CSV çıktılarını okuyarak
Phase 1 özet raporunu (Markdown) oluşturur.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
REPORT_DIR.mkdir(exist_ok=True)


def load_json(path):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    print("=" * 60)
    print("ADIM 5: Phase 1 Raporu Üretiliyor")
    print("=" * 60)

    col_result = load_json(OUT_DIR / "column_exploration_result.json")
    quality = load_json(OUT_DIR / "quality_summary.json")
    preproc = load_json(OUT_DIR / "preprocessing_summary.json")
    eda_stats = load_json(OUT_DIR / "eda_stats.json")
    splits_df = load_csv(OUT_DIR / "flight_splits.csv")
    meta_df = load_csv(OUT_DIR / "flight_meta.csv")
    quality_df = load_csv(OUT_DIR / "quality_report.csv")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# Phase 1 — Veri Analizi ve Ön İşleme Raporu")
    lines.append(f"\n**Üretim tarihi:** {now}")
    lines.append("\n---\n")

    # 1. Veri seti özeti
    lines.append("## 1. Veri Seti Özeti\n")
    lines.append(f"| Parametre | Değer |")
    lines.append(f"|---|---|")
    lines.append(f"| Toplam CSV dosyası | {quality.get('total_files', '?')} |")
    lines.append(f"| Geçerli uçuş | {quality.get('valid_flights', '?')} |")
    lines.append(f"| Reddedilen | {quality.get('rejected_flights', '?')} |")
    lines.append(f"| Toplam sütun (Holybro) | {col_result.get('total_columns', '?')} |")
    lines.append(f"| Benzersiz f-indeks grubu | {col_result.get('unique_f_indices', '?')} |")
    lines.append(f"| GPS leakage riski taşıyan sütun | {col_result.get('gps_leakage_count', '?')} |")
    lines.append("")

    # 2. Reddedilen uçuşlar
    rejected = quality.get("rejected_list", [])
    if rejected:
        lines.append("## 2. Reddedilen Uçuşlar\n")
        if not quality_df.empty:
            rej_df = quality_df[quality_df["status"] == "rejected"][["file", "row_count", "duration_s", "reject_reason"]]
            lines.append(rej_df.to_markdown(index=False))
        else:
            for r in rejected:
                lines.append(f"- {r}")
        lines.append("")

    # 3. Girdi sütunları
    lines.append("## 3. Model Girdi Sütunları (GPS Leakage Yok)\n")
    found_inputs = col_result.get("found_inputs", [])
    missing_inputs = col_result.get("missing_inputs", [])

    if found_inputs:
        lines.append(f"Bulunan ve kullanılacak sütunlar ({len(found_inputs)}):\n")
        lines.append("```")
        for c in found_inputs:
            lines.append(c)
        lines.append("```")
    if missing_inputs:
        lines.append(f"\n**Eksik sütunlar ({len(missing_inputs)}) — alternatif aranmalı:**")
        for c in missing_inputs:
            lines.append(f"- `{c}`")
    lines.append("")

    # 4. Ön işleme parametreleri
    lines.append("## 4. Ön İşleme Parametreleri\n")
    if preproc:
        lines.append(f"| Parametre | Değer |")
        lines.append(f"|---|---|")
        lines.append(f"| Hedef frekans | {preproc.get('target_hz', '?')} Hz |")
        lines.append(f"| Pencere uzunluğu | {preproc.get('sequence_length', '?')} adım |")
        lines.append(f"| Adım boyutu | {preproc.get('step_size', '?')} adım |")
        lines.append(f"| Feature sayısı | {preproc.get('n_features', '?')} |")
        lines.append(f"| İşlenen uçuş | {preproc.get('flights_processed', '?')} |")
        lines.append("")

    # 5. Split dağılımı
    lines.append("## 5. Eğitim / Doğrulama / Test Bölümü\n")
    if preproc and "split_sizes" in preproc:
        ss = preproc["split_sizes"]
        total = sum(ss.values())
        lines.append(f"| Küme | Örnek Sayısı | Oran |")
        lines.append(f"|---|---|---|")
        for s in ["train", "val", "test"]:
            n = ss.get(s, 0)
            lines.append(f"| {s.capitalize()} | {n:,} | %{100*n/total:.1f} |")
        lines.append("")

    if quality and "split" in quality:
        sp = quality["split"]
        lines.append(f"Uçuş bazlı split: Train={sp.get('train','?')} | Val={sp.get('val','?')} | Test={sp.get('test','?')}")
        lines.append("")

    # 6. Uçuş meta istatistikleri
    if not meta_df.empty:
        lines.append("## 6. Uçuş Süresi İstatistikleri\n")
        dur = meta_df["duration_s"]
        lines.append(f"| Stat | Değer |")
        lines.append(f"|---|---|")
        lines.append(f"| Ortalama | {dur.mean():.1f} s |")
        lines.append(f"| Medyan | {dur.median():.1f} s |")
        lines.append(f"| Min | {dur.min():.1f} s |")
        lines.append(f"| Maks | {dur.max():.1f} s |")
        lines.append(f"| Toplam | {dur.sum()/3600:.2f} saat |")
        lines.append("")

    # 7. Sensör istatistikleri
    if eda_stats:
        lines.append("## 7. Sensör İstatistikleri (3 Örnek Uçuştan)\n")
        lines.append(f"| Sütun | Ortalama | Std | Min | Maks |")
        lines.append(f"|---|---|---|---|---|")
        for col, s in eda_stats.items():
            if "error" not in s:
                lines.append(f"| `{col[:30]}` | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
        lines.append("")

    # 8. Grafikler
    lines.append("## 8. Üretilen Grafikler\n")
    plots_dir = OUT_DIR / "plots"
    if plots_dir.exists():
        plots = sorted(plots_dir.glob("*.png"))
        for p in plots:
            lines.append(f"- `outputs/plots/{p.name}`")
    else:
        lines.append("Grafik bulunamadı.")
    lines.append("")

    # 9. Sonraki adım
    lines.append("## 9. Phase 2 İçin Hazır Dosyalar\n")
    npy_files = list(OUT_DIR.glob("*.npy")) + list(OUT_DIR.glob("*.pkl")) + list(OUT_DIR.glob("*.npz"))
    lines.append(f"Toplam çıktı dosyası: {len(npy_files)}")
    lines.append("```")
    for f in sorted(OUT_DIR.glob("X_*.npy")):
        arr = np.load(f)
        lines.append(f"{f.name}: shape={arr.shape}")
    for f in sorted(OUT_DIR.glob("y_*.npy")):
        arr = np.load(f)
        lines.append(f"{f.name}: shape={arr.shape}")
    lines.append("scaler.pkl: StandardScaler (eğitim setinden fit)")
    lines.append("```")

    # Raporu yaz
    report_path = REPORT_DIR / "phase1_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nRapor kaydedildi: {report_path}")
    print("\nAdım 5 tamamlandı.")


if __name__ == "__main__":
    main()
