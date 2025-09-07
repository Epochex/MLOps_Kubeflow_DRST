#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, queue, threading
from collections import deque
from typing import List, Tuple, Optional, Dict

import numpy as np

from drst_common.config import (
    KAFKA_TOPIC, RESULT_DIR, TARGET_COL, BUCKET, MODEL_DIR
)
from drst_common.kafka_io import create_consumer, SENTINEL_HEADER_KEY, SENTINEL_HEADER_VAL, SENTINEL_FALLBACK_VALUE
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.minio_helper import save_np, save_bytes, s3
from drst_common.resource_probe import start as start_probe
from drst_common.utils import jensen_shannon_divergence, make_prob_hist

# ===== 你的设计：固定窗口/步长/阈值（无需额外 export）=====
DRIFT_WINDOW   = 300          # 窗口大小
EVAL_STRIDE    = 50           # 每进来 50 条评一次
JS_THR_A       = 0.40         # <0.40 不重训；[0.40,0.60) 用 A
JS_THR_B       = 0.60         # [0.60,0.75) 用 B
JS_THR_C       = 0.75         # ≥0.75    用 C
HIST_BINS      = 64
IDLE_TIMEOUT_S = 60
BASELINE_REFRESH_MODE = "on_retrain"  # 重训后用当前窗口刷新基线，便于后续收敛

pod_name = os.getenv("HOSTNAME", "monitor")

def _make_run_tag() -> str:
    for k in ("KFP_RUN_ID", "WORKFLOW_ID", "PIPELINE_RUN_ID", "POD_NAME", "HOSTNAME"):
        v = os.getenv(k)
        if v: return v
    return str(int(time.time()))
RUN_TAG   = _make_run_tag()
GROUP_ID  = f"cg-monitor-{RUN_TAG}"

q_data = queue.Queue()
data_partitions = 0
data_sentinel_seen = 0
sentinel_lock = threading.Lock()

def _load_feature_cols() -> List[str]:
    obj = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/feature_cols.json")
    cols = json.loads(obj["Body"].read().decode())
    if not isinstance(cols, list) or not cols:
        raise RuntimeError("feature_cols.json invalid")
    return [str(c) for c in cols]

FEATURE_COLS = _load_feature_cols()

baseline_buf: deque[np.ndarray] = deque(maxlen=DRIFT_WINDOW)
baseline_ready = False
window_buf:   deque[np.ndarray] = deque(maxlen=DRIFT_WINDOW)
label_buf:    deque[float]      = deque(maxlen=DRIFT_WINDOW)
_hist_range_per_feat: Optional[List[Tuple[float,float]]] = None

retrain_in_progress = False
lock_started_ts = 0.0

def _features_from_msg(msg) -> Tuple[np.ndarray, Optional[float]]:
    feats = []
    fdict = msg.get("features", None)
    # 兼容扁平字典（producer 当前就是扁平）
    if not isinstance(fdict, dict):
        fdict = msg
    for c in FEATURE_COLS:
        try:
            v = float(fdict.get(c, 0.0))
        except Exception:
            v = 0.0
        feats.append(v)
    y = msg.get(TARGET_COL, None)
    try: y = float(y) if y is not None else None
    except Exception: y = None
    return np.asarray(feats, dtype=np.float32), y

def _init_hist_ranges(baseline_arr: np.ndarray) -> None:
    global _hist_range_per_feat
    d = baseline_arr.shape[1]
    _hist_range_per_feat = []
    for j in range(d):
        mn = float(np.min(baseline_arr[:, j])); mx = float(np.max(baseline_arr[:, j]))
        if mn == mx:
            eps = (abs(mn) * 1e-6) or 1e-6
            mn, mx = mn - eps, mx + eps
        _hist_range_per_feat.append((mn, mx))

def _compute_js(baseline_arr: np.ndarray, window_arr: np.ndarray) -> float:
    assert _hist_range_per_feat is not None
    vals = []
    for j, rng in enumerate(_hist_range_per_feat):
        p = make_prob_hist(baseline_arr[:, j], bins=HIST_BINS, range=rng)
        q = make_prob_hist(window_arr[:, j],   bins=HIST_BINS, range=rng)
        vals.append(jensen_shannon_divergence(p, q))
    return float(np.mean(vals))

def _save_latest_batch(X: np.ndarray, y: Optional[np.ndarray]):
    cols = list(FEATURE_COLS)
    if y is not None:
        arr = np.concatenate([X, y.reshape(-1,1)], axis=1)
        cols.append(TARGET_COL)
    else:
        arr = X
    save_np(f"{RESULT_DIR}/latest_batch.npy", arr)
    save_bytes(f"{RESULT_DIR}/latest_batch.columns.json", json.dumps(cols, ensure_ascii=False, indent=2).encode(), "application/json")

def _grid_letter(js_val: float) -> str | None:
    # 固定阈值：<A 不重训；落入 [A,B)→A；[B,C)→B；≥C→C
    if js_val < JS_THR_A: return None
    if js_val < JS_THR_B: return "A"
    if js_val < JS_THR_C: return "B"
    return "C"

def _retrain_done_after(ts0: float) -> bool:
    key = f"{RESULT_DIR}/retrain_done.flag"
    try:
        meta = s3.head_object(Bucket=BUCKET, Key=key)
        return meta["LastModified"].timestamp() >= ts0
    except Exception:
        return False

def _is_sentinel_msg(msg) -> bool:
    try:
        for k, v in (msg.headers or []):
            if k == SENTINEL_HEADER_KEY and v == SENTINEL_HEADER_VAL:
                return True
    except Exception:
        pass
    try:
        if msg.value == SENTINEL_FALLBACK_VALUE:
            return True
    except Exception:
        pass
    try:
        obj = json.loads(msg.value)
        if isinstance(obj, dict) and obj.get("producer_done"):
            return True
    except Exception:
        pass
    return False

def _data_listener():
    global data_partitions, data_sentinel_seen
    cons = create_consumer(KAFKA_TOPIC, group_id=GROUP_ID)
    time.sleep(1.0)
    # 直接用 assignment 的分区数
    t0 = time.time()
    while not cons.assignment() and time.time() - t0 < 10:
        cons.poll(100)
    data_partitions = max(1, len(cons.assignment()))
    print(f"[monitor:{pod_name}] topic «{KAFKA_TOPIC}» partitions = {data_partitions}")
    touch_ready("monitor", pod_name)
    while True:
        polled = cons.poll(timeout_ms=1000)
        if not polled: continue
        for _, recs in polled.items():
            for msg in recs:
                if _is_sentinel_msg(msg):
                    with sentinel_lock:
                        data_sentinel_seen += 1
                        print(f"[monitor:{pod_name}] got sentinel {data_sentinel_seen}/{data_partitions}")
                    continue
                try:
                    v = json.loads(msg.value)
                    if not isinstance(v, dict): continue
                    q_data.put(v)
                except Exception:
                    continue

threading.Thread(target=_data_listener, daemon=True).start()

print(f"[monitor:{pod_name}] started…")
last_data_time = time.time()
stop_probe = start_probe("monitor")  # 资源采样

try:
    while True:
        # 解锁：等待 retrain 完成
        if retrain_in_progress and _retrain_done_after(lock_started_ts):
            retrain_in_progress = False
            print(f"[monitor:{pod_name}] retrain done detected → unlock")
            if BASELINE_REFRESH_MODE == "on_retrain":
                try:
                    X_win = np.vstack(window_buf) if len(window_buf) == DRIFT_WINDOW else None
                    if X_win is not None and X_win.size > 0:
                        baseline_buf.clear()
                        for row in X_win: baseline_buf.append(row.copy())
                        # 重置直方图范围
                        _init_hist_ranges(np.vstack(baseline_buf))
                        log_metric(component="monitor", event="baseline_refreshed")
                        print(f"[monitor:{pod_name}] baseline refreshed")
                except Exception as e:
                    print(f"[monitor:{pod_name}] baseline refresh failed: {e}")

        try:
            rec = q_data.get(timeout=1.0)
        except queue.Empty:
            with sentinel_lock:
                all_done = (data_sentinel_seen >= data_partitions) if data_partitions else False
            if all_done and q_data.empty():
                print(f"[monitor:{pod_name}] all sentinels seen; exit."); break
            if (time.time() - last_data_time) > IDLE_TIMEOUT_S and baseline_ready:
                print(f"[monitor:{pod_name}] idle >{IDLE_TIMEOUT_S}s; exit."); break
            continue

        last_data_time = time.time()
        feats_vec, y = _features_from_msg(rec)

        if not baseline_ready:
            baseline_buf.append(feats_vec)
            if len(baseline_buf) >= DRIFT_WINDOW:
                baseline_ready = True
                base_arr = np.vstack(baseline_buf)
                _init_hist_ranges(base_arr)
                log_metric(component="monitor", event="baseline_ready", train_rows=DRIFT_WINDOW)
                print(f"[monitor:{pod_name}] baseline ready with {DRIFT_WINDOW} rows")
            continue

        window_buf.append(feats_vec)
        if y is not None: label_buf.append(float(y))
        if len(window_buf) < DRIFT_WINDOW: continue
        if retrain_in_progress: continue

        # 每进来 STRIDE 条评一次
        if (len(window_buf) % EVAL_STRIDE) != 0: 
            continue

        X_base = np.vstack(baseline_buf)
        X_win  = np.vstack(window_buf)
        js_val = _compute_js(X_base, X_win)

        log_metric(component="monitor", event="js_tick",
                   js_val=round(js_val, 6),
                   thr_A=JS_THR_A, thr_B=JS_THR_B, thr_C=JS_THR_C,
                   window=DRIFT_WINDOW, eval_stride=EVAL_STRIDE)

        letter = _grid_letter(js_val)
        if letter is None: 
            continue

        y_latest = np.array(label_buf, dtype=np.float32) if len(label_buf) == len(window_buf) else None
        _save_latest_batch(X_win, y_latest)
        save_bytes(f"{RESULT_DIR}/retrain_grid.flag", letter.encode(), "text/plain")
        save_bytes(f"{RESULT_DIR}/last_js.txt", f"{js_val:.6f}\n".encode(), "text/plain")
        save_bytes(f"{RESULT_DIR}/retrain_lock.flag", b"", "text/plain")
        retrain_in_progress = True
        lock_started_ts = time.time()

        log_metric(component="monitor", event="drift_triggered", js_val=round(js_val, 6), grid=letter)
        print(f"[monitor:{pod_name}] drift triggered: js={js_val:.3f} → grid {letter}")
finally:
    sync_all_metrics_to_minio()
    write_kfp_metadata()
    stop_probe()
    print(f"[monitor:{pod_name}] metrics synced; bye.")
