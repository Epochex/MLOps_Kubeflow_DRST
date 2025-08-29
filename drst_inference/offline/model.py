#!/usr/bin/env python3
# drst_inference/offline/model.py
from __future__ import annotations
import torch
import torch.nn as nn

def _act_layer(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "tanh": return nn.Tanh()
    if name == "gelu": return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 64), act: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.config = {"in_dim": int(in_dim), "hidden": list(hidden), "act": act, "dropout": float(dropout)}
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), _act_layer(act)]
            if dropout and dropout > 0: layers.append(nn.Dropout(p=dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
