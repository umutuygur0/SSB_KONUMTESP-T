"""
Phase 1 — Tüm Adımları Sırayla Çalıştır
"""
import sys
import traceback
from pathlib import Path

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))


def run_step(module_name: str, step_label: str):
    print(f"\n{'#'*60}")
    print(f"  {step_label}")
    print(f"{'#'*60}")
    try:
        mod = __import__(module_name)
        if hasattr(mod, "main"):
            mod.main()
        print(f"  => {step_label} TAMAMLANDI\n")
        return True
    except Exception as e:
        print(f"  => HATA: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    steps = [
        ("01_column_explorer", "Adım 1: Sütun Keşfi"),
        ("02_quality_filter",  "Adım 2: Kalite Filtresi"),
        ("03_preprocessing",   "Adım 3: Ön İşleme"),
        ("04_eda_visualize",   "Adım 4: EDA Görselleştirme"),
        ("05_phase1_report",   "Adım 5: Rapor"),
    ]

    failed = []
    for module, label in steps:
        ok = run_step(module, label)
        if not ok:
            failed.append(label)
            print(f"  Uyarı: {label} başarısız, devam ediliyor...")

    print(f"\n{'='*60}")
    print("PHASE 1 TAMAMLANDI")
    if failed:
        print(f"Başarısız adımlar: {failed}")
    else:
        print("Tüm adımlar başarıyla tamamlandı.")
    print(f"Çıktılar: Phase1_Veri_Analizi/outputs/")
    print(f"Rapor   : Phase1_Veri_Analizi/report/phase1_report.md")
