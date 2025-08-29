#!/usr/bin/env python3
# drst_common/utils.py
from __future__ import annotations
import io
import math
from typing import Iterable, Tuple, Optional
import numpy as np

# --------- Torch model deserialization (with fault tolerance) ---------
def _bytes_to_model(b: bytes):
    """Fault-tolerant loading of PyTorch models: try torch.load first, then fallback to torch.jit.load."""
    import torch
    bio = io.BytesIO(b)
    try:
        return torch.load(bio, map_location="cpu")
    except Exception:
        bio.seek(0)
        try:
            return torch.jit.load(bio, map_location="cpu")
        except Exception as e:
            raise RuntimeError("Failed to load model bytes via torch.load / torch.jit.load") from e

# --------- Simple threshold-based accuracy ---------
def calculate_accuracy_within_threshold(y_pred: np.ndarray, y_true: np.ndarray, thr: float = 0.15) -> float:
    y_pred = np.asarray(y_pred, np.float32).ravel()
    y_true = np.asarray(y_true, np.float32).ravel()
    denom = np.maximum(np.abs(y_true), 1e-8)
    relerr = np.abs(y_pred - y_true) / denom
    return float(np.mean(relerr <= thr))

# --------- Probability histogram & Jensen-Shannon divergence ---------
def make_prob_hist(x: np.ndarray, bins: int = 64, range: Optional[Tuple[float,float]] = None) -> np.ndarray:
    x = np.asarray(x, np.float64).ravel()
    cnts, _ = np.histogram(x, bins=bins, range=range)
    p = cnts.astype(np.float64)
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / s
    # add tiny smoothing to avoid log(0)
    eps = 1e-12
    p = (p + eps) / (1.0 + eps * p.size)
    return p

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    # both p, q should be probability distributions
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
