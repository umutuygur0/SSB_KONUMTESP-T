"""
Paylaşılan model tanımları — GRU ve LSTM.
Hem eğitim hem değerlendirme scriptleri buradan import eder.
"""

import numpy as np
import torch
import torch.nn as nn

# Test uçuşlarında mag_field (6,7,8) ve baro_alt (10) eğitim dağılımından sapıyor:
#   mag1 test std = 37.6σ  (eğitimde ~1σ olmalı)  → DC + VAR farkı
#   baro test std =  4.6σ  (eğitimde ~0.7σ)
# Çözüm: her pencere içinde bu özellikler için z-score normalizasyonu uygula.
# Fiziksel gerekçe: model, mutlak mag/baro değerini değil pencere-içi DEĞİŞİMİ
# kullanır → lokal standardizasyon bilgiyi korur, kalibrasyon bağımlılığını kaldırır.
LOCAL_NORM_COLS = [6, 7, 8, 10]   # mag0, mag1, mag2, baro_alt


def _local_zscore(arr: np.ndarray, axis: int) -> np.ndarray:
    """arr'ı verilen axis boyunca z-score'a çevir; sabit sütunları 0 bırak."""
    mu = arr.mean(axis=axis, keepdims=True)
    sd = arr.std(axis=axis, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (arr - mu) / sd


def detrend_windows(X: np.ndarray) -> np.ndarray:
    """
    X: (N, SEQ_LEN, 12)  — ölçeklenmiş pencereler.
    LOCAL_NORM_COLS için her pencerede within-window z-score uygular.
    Dönüş: aynı shape, aynı dtype.
    """
    X = X.copy().astype(np.float32)
    X[:, :, LOCAL_NORM_COLS] = _local_zscore(
        X[:, :, LOCAL_NORM_COLS], axis=1
    )
    return X


def detrend_single(window: np.ndarray) -> np.ndarray:
    """
    window: (SEQ_LEN, 12) — tek pencere (GPS kesinti simülasyonu için).
    """
    window = window.copy().astype(np.float32)
    window[:, LOCAL_NORM_COLS] = _local_zscore(
        window[:, LOCAL_NORM_COLS], axis=0
    )
    return window


class GRUModel(nn.Module):
    """
    Tek yönlü GRU.  Input: (batch, 40, 12) → Output: (batch, 3)
    num_layers > 1 olduğunda layers arası dropout uygulanır.
    """
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


class LSTMModel(nn.Module):
    """
    Tek yönlü LSTM.  BiLSTM değil — gerçek zamanlı inference için uygun.
    Input: (batch, 40, 12) → Output: (batch, 3)
    """
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))
