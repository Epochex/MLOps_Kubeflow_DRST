#!/usr/bin/env python3
# drst_drift/monitor.py
# Fixed sliding window=300 + stride=50: every 50 new samples, compute JS using last 300.
# Trigger policy (改造版):
#   仅当 JS_now > JS_mean 且 JS_now 落入分段阈值时触发：
#     JS<0.40 → no retrain; [0.40,0.60) → A; [0.60,0.75) → B; ≥0.75 → C。
#   触发后写锁，直到重训完成前不再计算/触发。
from __future__ import annotations
import os, json, time, queue, threading
from collections import deque
from typing import List, Tuple, Optional

import numpy as np

from drst_common.config import (
    KAFKA_TOPIC, RESULT_DIR, TARGET_COL,
    DRIFT_WINDOW, EVAL_STRIDE,
    RETRAIN_JS_NO_RETRAIN, RETRAIN_JS_GRID_A, RETRAIN_JS_GRID_B,
)
from drst_common.kafka_io import create_consumer, partitions_count
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.minio_helper import save_np, save_bytes, s3
from drst_common.artefacts import load_selected_feats, load_scaler
from drst_common.utils import jensen_shannon_divergence, make_prob_hist
from drst_common.config import BUCKET

# ---------------- Parameters ----------------
IDLE_TIMEOUT_S    = int(os.getenv("IDLE_TIMEOUT_S", "60"))
HIST_BINS         = int(os.getenv("HIST_BINS", "64"))
pod_name = os.getenv("HOSTNAME", "monitor")

# ---------------- State & Queues ----------------
q_data = queue.Queue()
data_partitions = 0
data_sentinel_seen = 0
sentinel_lock = threading.Lock()
processed_total = 0

SELECTED_FEATS = load_selected_feats()
scaler = load_scaler()

baseline_buf: deque[np.ndarray] = deque(maxlen=DRIFT_WINDOW)  # Reference distribution (first filled baseline)
baseline_ready = False
window_buf:   deque[np.ndarray] = deque(maxlen=DRIFT_WINDOW)
label_buf:    deque[float]      = deque(maxlen=DRIFT_WINDOW)
_hist_range_per_feat: Optional[List[Tuple[float,float]]] = None

samples_since_eval = 0
js_history: List[float] = []

retrain_in_progress = False
lock_started_ts = 0.0

def _features_from_msg(msg) -> Tuple[np.ndarray, Optional[float]]:
    x = np.array([msg["features"].get(c, 0.0) for c in SELECTED_FEATS], dtype=np.float32)
    y = msg.get("label", None)
    try:
        y = float(y) if y is not None else None
    except Exception:
        y = None
    return x, y

def _compute_js(baseline_arr: np.ndarray, window_arr: np.ndarray) -> float:
    global _hist_range_per_feat
    n0, d = baseline_arr.shape
    n1, d2 = window_arr.shape
    assert d == d2, "dim mismatch"
    if _hist_range_per_feat is None:
        _hist_range_per_feat = []
        for j in range(d):
            mn = float(np.min(baseline_arr[:, j])); mx = float(np.max(baseline_arr[:, j]))
            if mn == mx:
                eps = (abs(mn) * 1e-6) or 1e-6
                mn, mx = mn - eps, mx + eps
            _hist_range_per_feat.append((mn, mx))
    vals = []
    for j in range(d):
        rng = _hist_range_per_feat[j]
        p = make_prob_hist(baseline_arr[:, j], bins=HIST_BINS, range=rng)
        q = make_prob_hist(window_arr[:, j],   bins=HIST_BINS, range=rng)
        vals.append(jensen_shannon_divergence(p, q))
    return float(np.mean(vals))

def _save_latest_batch(X: np.ndarray, y: Optional[np.ndarray]):
    cols = list(SELECTED_FEATS)
    if y is not None:
        arr = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        cols.append(TARGET_COL)
    else:
        arr = X
    save_np(f"{RESULT_DIR}/latest_batch.npy", arr)
    save_bytes(f"{RESULT_DIR}/latest_batch.columns.json",
               json.dumps(cols, ensure_ascii=False, indent=2).encode(),
               "application/json")

def _grid_letter(js_val: float) -> str | None:
    if js_val < RETRAIN_JS_NO_RETRAIN:
        return None
    if js_val < RETRAIN_JS_GRID_A:
        return "A"
    if js_val < RETRAIN_JS_GRID_B:
        return "B"
    return "C"

def _retrain_done_after(ts0: float) -> bool:
    key = f"{RESULT_DIR}/retrain_done.flag"
    try:
        meta = s3.head_object(Bucket=BUCKET, Key=key)
        return meta["LastModified"].timestamp() >= ts0
    except Exception:
        return False

# ---------------- Listener thread ----------------
def _data_listener():
    global data_partitions, data_sentinel_seen
    cons = create_consumer(KAFKA_TOPIC, group_id="cg-monitor-data")
    time.sleep(1.0)
    data_partitions = partitions_count(cons, KAFKA_TOPIC)
    print(f"[monitor:{pod_name}] topic «{KAFKA_TOPIC}» partitions = {data_partitions}")
    touch_ready("monitor", pod_name)

    for msg in cons:
        v = msg.value
        if isinstance(v, dict) and v.get("producer_done"):
            with sentinel_lock:
                data_sentinel_seen += 1
                print(f"[monitor:{pod_name}] got sentinel {data_sentinel_seen}/{data_partitions}")
            continue
        q_data.put(v)

threading.Thread(target=_data_listener, daemon=True).start()

print(f"[monitor:{pod_name}] started…")
last_data_time = time.time()

while True:
    # 解锁检查：重训完成？
    if retrain_in_progress and _retrain_done_after(lock_started_ts):
        retrain_in_progress = False
        print(f"[monitor:{pod_name}] retrain done detected → unlock")

    try:
        rec = q_data.get(timeout=1.0)
    except queue.Empty:
        with sentinel_lock:
            all_done = (data_sentinel_seen >= data_partitions) if data_partitions else False
        if all_done and q_data.empty():
            print(f"[monitor:{pod_name}] all sentinels seen; exit.")
            break
        if (time.time() - last_data_time) > IDLE_TIMEOUT_S and baseline_ready:
            print(f"[monitor:{pod_name}] idle >{IDLE_TIMEOUT_S}s; exit.")
            break
        continue

    last_data_time = time.time()

    feats, y = _features_from_msg(rec)
    feats_scaled = scaler.transform(feats.reshape(1, -1)).ravel()

    # 建立基线（首 300 个样本）
    if not baseline_ready:
        baseline_buf.append(feats_scaled)
        if len(baseline_buf) >= DRIFT_WINDOW:
            baseline_ready = True
            log_metric(component="monitor", event="baseline_ready",
                       train_rows=DRIFT_WINDOW)
            print(f"[monitor:{pod_name}] baseline ready with {DRIFT_WINDOW} rows")
        continue

    # 基线已就绪，进入滑窗
    window_buf.append(feats_scaled)
    if y is not None:
        label_buf.append(float(y))
    if len(window_buf) < DRIFT_WINDOW:
        continue

    # 重训锁定中：暂停评估与触发
    if retrain_in_progress:
        continue

    samples_since_eval += 1
    if samples_since_eval < EVAL_STRIDE:
        continue
    samples_since_eval = 0

    X_base = np.vstack(baseline_buf)
    X_win  = np.vstack(window_buf)
    js_val = _compute_js(X_base, X_win)
    js_mean = float(np.mean(js_history)) if js_history else 0.0
    js_history.append(js_val)

    log_metric(component="monitor", event="js_tick",
               js_val=round(js_val, 6),
               js_mean=round(js_mean, 6),
               window=DRIFT_WINDOW, eval_stride=EVAL_STRIDE)

    # 仅当“相对历史抬升”且“绝对阈值分段”同时满足时触发
    if not (js_val > js_mean):
        continue

    letter = _grid_letter(js_val)
    if letter is None:
        continue  # 绝对阈值不足

    # 触发：保存窗口数据 + 写网格标志 + 上锁
    y_latest = np.array(label_buf, dtype=np.float32) if len(label_buf) == len(window_buf) else None
    _save_latest_batch(X_win, y_latest)
    save_bytes(f"{RESULT_DIR}/retrain_grid.flag", letter.encode(), "text/plain")
    save_bytes(f"{RESULT_DIR}/last_js.txt", f"{js_val:.6f}\n".encode(), "text/plain")
    save_bytes(f"{RESULT_DIR}/retrain_lock.flag", b"", "text/plain")
    retrain_in_progress = True
    lock_started_ts = time.time()

    log_metric(component="monitor", event="drift_triggered",
               js_val=round(js_val, 6), js_mean=round(js_mean, 6), grid=letter)

# 收尾
sync_all_metrics_to_minio()
write_kfp_metadata()
print(f"[monitor:{pod_name}] metrics synced; bye.")
