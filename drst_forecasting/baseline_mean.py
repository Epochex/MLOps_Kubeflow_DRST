#!/usr/bin/env python3
# drst_forecasting/baseline_mean.py
from __future__ import annotations
import numpy as np

class BaselineMean:
    def __init__(self, window: int = 10):
        self.window = int(window)
        self.buf: list[float] = []

    def update(self, y: float) -> float:
        self.buf.append(float(y))
        if len(self.buf) > self.window:
            self.buf = self.buf[-self.window:]
        return float(np.mean(self.buf))

    def predict(self, arr: np.ndarray) -> np.ndarray:
        out = []
        for v in arr:
            out.append(self.update(float(v)))
        return np.array(out, dtype=np.float32)
