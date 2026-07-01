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


class AttentionGRUModel(nn.Module):
    """
    GRU + Self-Attention pooling.
    Son hidden state yerine tüm timestep çıktıları üzerinde öğrenilebilir
    ağırlıklı ortalama alınır — hangi zaman diliminin önemli olduğunu model öğrenir.
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
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn    = nn.Linear(hidden_size, 1)   # attention scoring: H → 1
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                           # (B, T, H)
        out    = self.dropout(out)
        scores  = self.attn(out).squeeze(-1)           # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        context = (weights.unsqueeze(-1) * out).sum(dim=1)  # (B, H)
        return self.fc(context)


class CNNModel(nn.Module):
    """
    1D Dilated CNN — RNN'den farklı inductive bias.
    Dilated causal convolutions ile uzun menzilli bağımlılıklar.
    Input: (batch, 40, 12) → Output: (batch, 3)
    """
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        dropout: float = 0.2,
        output_size: int = 3,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            # (B, 12, T) → (B, 64, T) dilation=1
            nn.Sequential(
                nn.Conv1d(input_size, 64, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(), nn.Dropout(dropout),
            ),
            # (B, 64, T) → (B, 128, T) dilation=2
            nn.Sequential(
                nn.Conv1d(64, hidden_size, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(), nn.Dropout(dropout),
            ),
            # (B, 128, T) → (B, 128, T) dilation=4
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=8, dilation=4),
                nn.ReLU(), nn.Dropout(dropout),
            ),
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, F, T)
        for conv in self.convs:
            x = conv(x)[..., :40]      # trim to original T=40
        x = x.mean(dim=2)              # global average pooling → (B, H)
        return self.fc(x)


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
