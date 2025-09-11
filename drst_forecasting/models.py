# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/models.py
from __future__ import annotations
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    Input:  [B, T, D]  (lookback length T, feature dim D)
    Output: [B, H]     (horizon H)
    """
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1, horizon: int = 1, dropout: float = 0.0):
        super().__init__()
        self.config = {
            "in_dim": int(in_dim),
            "hidden": int(hidden),
            "num_layers": int(num_layers),
            "horizon": int(horizon),
            "dropout": float(dropout)
        }
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        # x: [B, T, D]
        y, (h, c) = self.lstm(x)        # y: [B, T, H], take last step
        last = y[:, -1, :]              # [B, hidden]
        out = self.head(last)           # [B, horizon]
        return out
