#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, queue, threading
from collections import deque
from typing import List, Tuple, Optional, Dict

import numpy as np

from drst_common.config import (
    KAFKA_TOPIC, RESULT_DIR, TARGET_COL, DRIFT_WINDOW, EVAL_STRIDE, BUCKET, MODEL_DIR
)
from drst_common.kafka_io import create_consumer, partitions_count
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.minio_helper import save_np, save_bytes, s3
from drst_common.resource_probe import start as start_probe
from drst_common.utils import jensen_shannon_divergence, make_prob_hist

IDLE_TIMEOUT_S       = int(os.getenv("IDLE_TIMEOUT_S", "60"))
HIST_BINS            = int(os.getenv("HIST_BINS", "64"))
JS_CALIB_SAMPLES     = int(os.getenv("JS_CALIB_SAMPLES", "400"))
JS_QUANTILES_RAW     = os.getenv("JS_QUANTILES", "0.90,0.97,0.995")
BASELINE_REFRESH_MODE= os.getenv("BASELINE_REFRESH_MODE", "on_retrain").lower()
pod_name = os.getenv("HOSTNAME", "monitor")

def _make_run_tag() -> str:
    for k in ("KFP_RUN_ID", "WORKFLOW_ID", "PIPELINE_RUN_ID", "POD_NAME", "HOSTNAME"):
        v = os.getenv(k)
        if v: return v
    return str(int(time.time()))
RUN_TAG   = _make_run_tag()
GROUP_ID  = f"cg-monitor-{RUN_TAG}"

def _parse_quantiles(s: str) -> Tuple[float, float, float]:
    try:
        q = [float(x.strip()) for x in s.split(",")]
        q = (q + [0.9, 0.97, 0.995])[:3]
        qa, qb, qc = sorted([max(0.0, min(q[0], 0.999)), max(0.0, min(q[1], 0.999)), max(0.0, min(q[2], 0.999))])
        return qa, qb, qc
    except Exception:
        return 0.90, 0.97, 0.995

Q_A, Q_B, Q_C = _parse_quantiles(JS_QUANTILES_RAW)

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

samples_since_eval = 0
_js_thr: Dict[str, float] = {}
retrain_in_progress = False
lock_started_ts = 0.0

def _features_from_msg(msg) -> Tuple[np.ndarray, Optional[float]]:
    feats = []
    fdict = msg.get("features", {}) or {}
    for c in FEATURE_COLS:
        try:
            v = float(fdict.get(c, 0.0))
        except Exception:
            v = 0.0
        feats.append(v)
    y = msg.get("label", None)
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

def _bootstrap_calibrate(baseline_arr: np.ndarray) -> Dict[str, float]:
    _init_hist_ranges(baseline_arr)
    js_vals = []
    n = baseline_arr.shape[0]
    rng = np.random.default_rng(12345)
    for _ in range(max(50, JS_CALIB_SAMPLES)):
        idx = rng.integers(0, n, size=n)
        boot = baseline_arr[idx]
        js_vals.append(_compute_js(baseline_arr, boot))
    js_vals = np.sort(np.asarray(js_vals, np.float64))
    thr_A = float(np.quantile(js_vals, Q_A))
    thr_B = float(np.quantile(js_vals, Q_B))
    thr_C = float(np.quantile(js_vals, Q_C))
    thr_B = max(thr_B, thr_A + 1e-12)
    thr_C = max(thr_C, thr_B + 1e-12)
    payload = {"quantiles":{"A":Q_A,"B":Q_B,"C":Q_C},"thresholds":{"A":thr_A,"B":thr_B,"C":thr_C},"bins":HIST_BINS,"window":DRIFT_WINDOW,"samples":JS_CALIB_SAMPLES}
    save_bytes(f"{RESULT_DIR}/js_calib.json", json.dumps(payload, ensure_ascii=False, indent=2).encode(),"application/json")
    log_metric(component="monitor", event="js_calibrated", thr_A=round(thr_A,6), thr_B=round(thr_B,6), thr_C=round(thr_C,6),
               qA=Q_A, qB=Q_B, qC=Q_C, window=DRIFT_WINDOW, bins=HIST_BINS, samples=JS_CALIB_SAMPLES)
    return {"A":thr_A,"B":thr_B,"C":thr_C}

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
    if not _js_thr: return None
    if js_val < _js_thr["A"]: return None
    if js_val < _js_thr["B"]: return "A"
    if js_val < _js_thr["C"]: return "B"
    return "C"

def _retrain_done_after(ts0: float) -> bool:
    key = f"{RESULT_DIR}/retrain_done.flag"
    try:
        meta = s3.head_object(Bucket=BUCKET, Key=key)
        return meta["LastModified"].timestamp() >= ts0
    except Exception:
        return False

def _data_listener():
    global data_partitions, data_sentinel_seen
    cons = create_consumer(KAFKA_TOPIC, group_id=GROUP_ID)
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
stop_probe = start_probe("monitor")  # 资源采样

try:
    while True:
        if retrain_in_progress and _retrain_done_after(lock_started_ts):
            retrain_in_progress = False
            print(f"[monitor:{pod_name}] retrain done detected → unlock")
            if BASELINE_REFRESH_MODE == "on_retrain":
                try:
                    X_win = np.vstack(window_buf) if len(window_buf) == DRIFT_WINDOW else None
                    if X_win is not None and X_win.size > 0:
                        baseline_buf.clear()
                        for row in X_win: baseline_buf.append(row.copy())
                        base_arr = np.vstack(baseline_buf)
                        _js_thr = _bootstrap_calibrate(base_arr)
                        log_metric(component="monitor", event="baseline_refreshed")
                        print(f"[monitor:{pod_name}] baseline refreshed & thresholds recalibrated")
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
                _js_thr = _bootstrap_calibrate(base_arr)
                log_metric(component="monitor", event="baseline_ready", train_rows=DRIFT_WINDOW)
                print(f"[monitor:{pod_name}] baseline ready with {DRIFT_WINDOW} rows")
            continue

        window_buf.append(feats_vec)
        if y is not None: label_buf.append(float(y))
        if len(window_buf) < DRIFT_WINDOW: continue
        if retrain_in_progress: continue

        samples_since_eval += 1
        if samples_since_eval < EVAL_STRIDE: continue
        samples_since_eval = 0

        X_base = np.vstack(baseline_buf)
        X_win  = np.vstack(window_buf)
        js_val = _compute_js(X_base, X_win)

        log_metric(component="monitor", event="js_tick",
                   js_val=round(js_val, 6),
                   thr_A=_js_thr.get("A",""), thr_B=_js_thr.get("B",""), thr_C=_js_thr.get("C",""),
                   window=DRIFT_WINDOW, eval_stride=EVAL_STRIDE)

        letter = _grid_letter(js_val)
        if letter is None: continue

        y_latest = np.array(label_buf, dtype=np.float32) if len(label_buf) == len(window_buf) else None
        _save_latest_batch(X_win, y_latest)
        save_bytes(f"{RESULT_DIR}/retrain_grid.flag", letter.encode(), "text/plain")
        save_bytes(f"{RESULT_DIR}/last_js.txt", f"{js_val:.6f}\n".encode(), "text/plain")
        save_bytes(f"{RESULT_DIR}/retrain_lock.flag", b"", "text/plain")
        retrain_in_progress = True
        lock_started_ts = time.time()

        log_metric(component="monitor", event="drift_triggered", js_val=round(js_val, 6), grid=letter)
finally:
    sync_all_metrics_to_minio()
    write_kfp_metadata()
    stop_probe()  # 上传 results/monitor_resources.csv
    print(f"[monitor:{pod_name}] metrics synced; bye.")
