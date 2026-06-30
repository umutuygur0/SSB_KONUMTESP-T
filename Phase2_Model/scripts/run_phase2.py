"""
Phase 2 — Tüm Adımları Sırayla Çalıştır
Kullanım: python run_phase2.py [--skip-train]
  --skip-train : model ağırlıkları zaten mevcutsa eğitimi atla
"""

import sys
import argparse
import time
from pathlib import Path

# scripts/ dizinine path ekle
sys.path.insert(0, str(Path(__file__).parent))

PHASE2_OUT = Path(__file__).resolve().parent.parent / "outputs"


def sep(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="model .pt dosyaları varsa eğitimi atla")
    args = parser.parse_args()

    t_total = time.time()

    # ── Adım 1: Dead Reckoning Baseline ──────────────────────────────
    sep("ADIM 1 / 5 — Dead Reckoning Baseline")
    import importlib
    dr_mod = importlib.import_module("01_baseline_dead_reckoning")
    dr_mod.main()

    # ── Adım 2: GRU Eğitimi ───────────────────────────────────────────
    gru_pt = PHASE2_OUT / "best_gru.pt"
    if args.skip_train and gru_pt.exists():
        sep("ADIM 2 / 5 — GRU (atlandı, .pt mevcut)")
    else:
        sep("ADIM 2 / 5 — GRU Eğitimi")
        gru_mod = importlib.import_module("02_train_gru")
        gru_mod.train()

    # ── Adım 3: LSTM Eğitimi ─────────────────────────────────────────
    lstm_pt = PHASE2_OUT / "best_lstm.pt"
    if args.skip_train and lstm_pt.exists():
        sep("ADIM 3 / 5 — LSTM (atlandı, .pt mevcut)")
    else:
        sep("ADIM 3 / 5 — LSTM Eğitimi")
        lstm_mod = importlib.import_module("03_train_lstm")
        lstm_mod.train()

    # ── Adım 4: Değerlendirme & GPS Kesinti ───────────────────────────
    sep("ADIM 4 / 5 — Değerlendirme & GPS Kesinti Deneyi")
    eval_mod = importlib.import_module("04_evaluate")
    eval_mod.main()

    # ── Adım 5: Rapor ────────────────────────────────────────────────
    sep("ADIM 5 / 5 — Markdown Raporu")
    rep_mod = importlib.import_module("05_phase2_report")
    rep_mod.main()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  PHASE 2 TAMAMLANDI  ({elapsed/60:.1f} dakika)")
    print(f"  Çıktı dizini: {PHASE2_OUT}")
    print(f"  Rapor: {PHASE2_OUT.parent / 'report' / 'phase2_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
