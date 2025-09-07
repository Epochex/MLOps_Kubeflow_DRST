#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, io, json, random, collections
from typing import Deque, Optional, List, Tuple
import numpy as np

from drst_common.kafka_io import create_consumer, is_sentinel, partitions_for_topic
from drst_common.minio_helper import s3, save_bytes
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.resource_probe import start as start_probe
from drst_common.config import (
    BUCKET, MODEL_DIR, RESULT_DIR, KAFKA_TOPIC, TARGET_COL,
    DRIFT_WINDOW, EVAL_STRIDE, HIST_BINS,
    JS_THR_A, JS_THR_B, JS_THR_C,
    MONITOR_WAIT_RETRAIN, MONITOR_IDLE_TIMEOUT_S, MAX_WALL_SECS,
    WAIT_FEATURES_SECS
)

RUN_TAG = os.getenv("KFP_RUN_ID") or os.getenv("PIPELINE_RUN_ID") or "drst"
TOPIC   = os.getenv("KAFKA_TOPIC", KAFKA_TOPIC)

LOCK_KEY  = f"{RESULT_DIR}/retrain_lock.flag"
GRID_KEY  = f"{RESULT_DIR}/retrain_grid.flag"
BATCH_KEY = f"{RESULT_DIR}/latest_batch.npy"
COLS_KEY  = f"{RESULT_DIR}/latest_batch.columns.json"
DONE_KEY  = f"{RESULT_DIR}/retrain_done.flag"

def _exists(key:str)->bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False

def _last_modified_ts(key:str) -> Optional[float]:
    try:
        meta = s3.head_object(Bucket=BUCKET, Key=key)
        return meta["LastModified"].timestamp()
    except Exception:
        return None

def _save_latest_batch(arr: np.ndarray, cols: List[str]):
    with io.BytesIO() as f:
        np.save(f, arr.astype(np.float32), allow_pickle=False)
        save_bytes(BATCH_KEY, f.getvalue(), "application/octet-stream")
    save_bytes(COLS_KEY, json.dumps(cols, ensure_ascii=False).encode("utf-8"), "application/json")

def _signal_retrain(grid: str):
    save_bytes(LOCK_KEY, b"1\n", "text/plain")
    save_bytes(GRID_KEY, grid.encode(), "text/plain")

def _retrain_done_after(ts0: float) -> bool:
    ts = _last_modified_ts(DONE_KEY)
    return (ts is not None) and (ts >= ts0)

def _load_feature_cols(wait_secs:int) -> List[str]:
    key = f"{MODEL_DIR}/feature_cols.json"
    deadline = time.time() + wait_secs
    last_err = None
    while time.time() < deadline:
        try:
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            cols = json.loads(raw.decode("utf-8"))
            if isinstance(cols, list) and cols:
                return [str(c) for c in cols]
        except Exception as e:
            last_err = e
        time.sleep(2)
    raise RuntimeError(f"wait feature_cols.json timeout: {last_err}")

def _parse_record(value_bytes: bytes, feat_cols: List[str]) -> Tuple[np.ndarray, Optional[float]]:
    obj = json.loads(value_bytes.decode("utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("features"), dict):
        feats = obj["features"]
        label_val = obj.get("label", obj.get(TARGET_COL))
    else:
        feats = obj if isinstance(obj, dict) else {}
        label_val = (obj.get(TARGET_COL) if isinstance(obj, dict) else None)
    vec = np.asarray([float(feats.get(c, 0.0)) for c in feat_cols], dtype=np.float32)
    try:
        y = float(label_val) if label_val is not None else None
    except Exception:
        y = None
    return vec, y

def _compute_js(baseline: np.ndarray, window: np.ndarray, bins:int) -> float:
    eps = 1e-12
    vals = []
    for j in range(baseline.shape[1]):
        a = baseline[:, j]; b = window[:, j]
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.max(), b.max()))
        if hi <= lo:
            continue
        P, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        Q, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        P = P.astype(np.float64) + eps
        Q = Q.astype(np.float64) + eps
        P /= P.sum(); Q /= Q.sum()
        M = 0.5 * (P + Q)
        kl_pm = float(np.sum(P * np.log(P / M)))
        kl_qm = float(np.sum(Q * np.log(Q / M)))
        js = 0.5 * (kl_pm + kl_qm)
        vals.append(js)
    return float(np.mean(vals)) if vals else 0.0

def _poll_one(consumer, timeout_ms: int = 1000):
    """统一的 poll 解包：返回按时间顺序的 ConsumerRecord 列表"""
    polled = consumer.poll(timeout_ms=timeout_ms)
    if not polled:
        return []
    out = []
    for _tp, recs in polled.items():
        if recs:
            out.extend(recs)
    return out

def main():
    pod = os.getenv("HOSTNAME", "monitor")
    feat_cols = _load_feature_cols(WAIT_FEATURES_SECS)

    consumer = create_consumer(TOPIC, group_id=f"cg-monitor-{RUN_TAG}")
    parts = partitions_for_topic(TOPIC) or set()

    print(f"[monitor:{pod}] started…", flush=True)
    print(f"[monitor:{pod}] topic «{TOPIC}» partitions = {len(parts)}", flush=True)

    stop_probe = start_probe("monitor")

    baseline_feats: Optional[np.ndarray] = None
    window_feats: Deque[np.ndarray] = collections.deque(maxlen=DRIFT_WINDOW)
    window_labels: Deque[Optional[float]] = collections.deque(maxlen=DRIFT_WINDOW)

    t_start = time.time()
    last_rx = time.time()
    all_sentinels = 0
    retrain_in_progress = False
    lock_started_ts: Optional[float] = None

    # ---- 1) 基线预热（用 poll，避免 StopIteration）----
    while True:
        msgs = _poll_one(consumer, timeout_ms=1000)
        if not msgs:
            # 超时检查
            if time.time() - t_start > MAX_WALL_SECS:
                print(f"[monitor:{pod}] max wall exceeded while warming baseline; exit.", flush=True)
                sync_all_metrics_to_minio(); stop_probe(); return
            continue

        for msg in msgs:
            last_rx = time.time()
            if is_sentinel(msg):
                all_sentinels += 1
                continue
            try:
                x, y = _parse_record(msg.value, feat_cols)
            except Exception:
                continue
            window_feats.append(x)
            window_labels.append(y)
            if len(window_feats) >= DRIFT_WINDOW:
                baseline_feats = np.stack(list(window_feats), axis=0)
                print(f"[monitor:{pod}] baseline ready with {baseline_feats.shape[0]} rows", flush=True)
                # 清空窗口（后续重新累计当前窗口）
                window_feats.clear(); window_labels.clear()
                break
        if baseline_feats is not None:
            break

    # ---- 2) 监控主循环 ----
    stride_since_eval = 0
    while True:
        # 主动超时/墙钟
        if (time.time() - last_rx) > MONITOR_IDLE_TIMEOUT_S:
            if MONITOR_WAIT_RETRAIN and retrain_in_progress and not _retrain_done_after(lock_started_ts or 0.0):
                time.sleep(1.0); continue
            print(f"[monitor:{pod}] idle > {MONITOR_IDLE_TIMEOUT_S}s; exit.", flush=True)
            break
        if time.time() - t_start > MAX_WALL_SECS:
            if MONITOR_WAIT_RETRAIN and retrain_in_progress and not _retrain_done_after(lock_started_ts or 0.0):
                time.sleep(1.0); continue
            print(f"[monitor:{pod}] max wall exceeded; exit.", flush=True)
            break

        msgs = _poll_one(consumer, timeout_ms=500)
        if not msgs:
            time.sleep(0.05)
            continue

        for msg in msgs:
            if is_sentinel(msg):
                all_sentinels += 1
                # 所有分区都收尾
                if all_sentinels >= max(1, len(parts)):
                    if MONITOR_WAIT_RETRAIN and retrain_in_progress and not _retrain_done_after(lock_started_ts or 0.0):
                        # 等重训完成
                        time.sleep(1.0); continue
                    print(f"[monitor:{pod}] all sentinels seen; exit.", flush=True)
                    sync_all_metrics_to_minio(); stop_probe()
                    print(f"[monitor:{pod}] metrics synced; bye.", flush=True)
                    return
                continue

            last_rx = time.time()
            try:
                x, y = _parse_record(msg.value, feat_cols)
            except Exception:
                continue
            window_feats.append(x)
            window_labels.append(y)
            stride_since_eval += 1

            if retrain_in_progress and MONITOR_WAIT_RETRAIN:
                continue

            if len(window_feats) >= DRIFT_WINDOW and stride_since_eval >= EVAL_STRIDE:
                stride_since_eval = 0
                cur_feats = np.stack(list(window_feats)[-DRIFT_WINDOW:], axis=0)
                js = _compute_js(baseline_feats, cur_feats, HIST_BINS)

                grid = None
                if js >= JS_THR_C:   grid = "C"
                elif js >= JS_THR_B: grid = "B"
                elif js >= JS_THR_A: grid = "A"

                if grid:
                    print(f"[monitor:{pod}] drift triggered: js={js:.3f} → grid {grid}", flush=True)
                    log_metric(component="monitor", event="drift", js=round(js, 6), grid=grid)

                    # 只收集“有标签”的行作为重训样本
                    lbls = list(window_labels)[-DRIFT_WINDOW:]
                    mask = np.array([ (l is not None) and np.isfinite(l) for l in lbls ], dtype=bool)
                    if mask.any():
                        feats_labeled = cur_feats[mask]
                        y_arr = np.array([float(lbls[i]) for i in range(len(lbls)) if mask[i]], dtype=np.float32)
                        batch = np.column_stack([feats_labeled, y_arr])
                        _save_latest_batch(batch, cols=[*feat_cols, TARGET_COL])
                    else:
                        # 没有标签也要触发锁，重训侧会优雅跳过
                        _save_latest_batch(cur_feats, cols=feat_cols)

                    _signal_retrain(grid)
                    retrain_in_progress = True
                    lock_started_ts = time.time()
                    continue

                log_metric(component="monitor", event="js_tick", js=round(js, 6))

            if retrain_in_progress and _retrain_done_after(lock_started_ts or 0.0):
                print(f"[monitor:{pod}] retrain done detected → unlock + baseline refresh", flush=True)
                retrain_in_progress = False
                # 刷新基线（按策略；这里采用在重训后刷新）
                if len(window_feats) >= DRIFT_WINDOW:
                    baseline_feats = np.stack(list(window_feats)[-DRIFT_WINDOW:], axis=0)
                log_metric(component="monitor", event="retrain_done_seen", ts=time.time())

    sync_all_metrics_to_minio()
    stop_probe()
    print(f"[monitor:{pod}] metrics synced; bye.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
