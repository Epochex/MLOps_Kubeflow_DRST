#!/usr/bin/env python3
# drst_forecasting/lstm_placeholder.py
from __future__ import annotations
import torch
import torch.nn as nn

class TinyLSTMReg(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (N, T, D)
        y, _ = self.lstm(x)
        out = self.head(y[:, -1, :])
        return out
