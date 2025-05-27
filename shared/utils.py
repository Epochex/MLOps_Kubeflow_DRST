import numpy as np
from scipy.spatial.distance import jensenshannon

def calculate_accuracy_within_threshold(y_true, y_pred, thr=0.15):
    dist = np.abs(y_true - y_pred) / y_true
    return (dist <= thr).mean() * 100

def js_divergence(p, q):
    return jensenshannon(p, q) ** 2

def filter_predictions(y_true, y_pred, threshold=0.1, accurate=True):
    err  = np.abs(y_true - y_pred) / y_true
    mask = err <= threshold if accurate else err > threshold
    return y_pred[mask], y_true[mask], np.where(mask)[0]

def filter_min(y_true, y_pred, min_value=500):     # ← 参数名 min_value
    mask = y_true >= min_value                     # ← 这里也用 min_value
    return y_pred[mask], y_true[mask], np.where(mask)[0]
