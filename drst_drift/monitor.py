#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, io, json, random, collections
from typing import Deque, Optional, List, Tuple
import numpy as np

from drst_common.kafka_io import create_consumer, is_sentinel, partitions_for_topic
from drst_common.minio_helper import s3, save_bytes
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.resource_probe import start as start_probe, update_extra
from drst_common.config import (
    BUCKET, MODEL_DIR, RESULT_DIR, KAFKA_TOPIC, TARGET_COL,
    DRIFT_WINDOW, EVAL_STRIDE, HIST_BINS,
    JS_THR_A, JS_THR_B, JS_THR_C,
    MONITOR_SIGNAL_INFER_PAUSE, MONITOR_IDLE_TIMEOUT_S, MAX_WALL_SECS,
    WAIT_FEATURES_SECS, PAUSE_INFER_KEY
)

RUN_TAG = os.getenv("KFP_RUN_ID") or os.getenv("PIPELINE_RUN_ID") or "drst"
TOPIC   = os.getenv("KAFKA_TOPIC", KAFKA_TOPIC)

LOCK_KEY  = f"{RESULT_DIR}/retrain_lock.flag"
GRID_KEY  = f"{RESULT_DIR}/retrain_grid.flag"
BATCH_KEY = f"{RESULT_DIR}/latest_batch.npy"
BATCH_COL = f"{RESULT_DIR}/latest_batch.columns.json"
DONE_KEY  = f"{RESULT_DIR}/retrain_done.flag"

# 冷却期（秒），默认 10，可用环境变量 RETRAIN_COOLDOWN_S 覆盖
RETRAIN_COOLDOWN_S = int(os.getenv("RETRAIN_COOLDOWN_S", "10") or 10)

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

def _save_latest_batch_xy(x: np.ndarray, y: np.ndarray, cols: List[str]):
    assert x.ndim == 2 and y.ndim == 1 and x.shape[0] == y.shape[0]
    arr = np.concatenate([x.astype(np.float32), y.astype(np.float32).reshape(-1,1)], axis=1)
    with io.BytesIO() as f:
        np.save(f, arr, allow_pickle=False)
        save_bytes(BATCH_KEY, f.getvalue(), "application/octet-stream")
    save_bytes(BATCH_COL, json.dumps(cols + [TARGET_COL]).encode("utf-8"), "application/json")

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

def _xy_from_record(raw: bytes, feat_cols: List[str]) -> Tuple[np.ndarray, Optional[float]]:
    obj = json.loads(raw.decode("utf-8"))
    feats = obj.get("features", {})
    vec = [float(feats.get(c, 0.0)) for c in feat_cols]
    y = obj.get("label", None)
    y = float(y) if y is not None else None
    return np.asarray(vec, dtype=np.float32), y

def _compute_js(baseline_x: np.ndarray, window_x: np.ndarray, bins:int) -> float:
    eps = 1e-12
    vals = []
    for j in range(baseline_x.shape[1]):
        a = baseline_x[:, j]; b = window_x[:, j]
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

def main():
    pod = os.getenv("HOSTNAME", "monitor")
    feat_cols = _load_feature_cols(WAIT_FEATURES_SECS)
    consumer = create_consumer(TOPIC, group_id=f"cg-monitor-{RUN_TAG}")
    parts = partitions_for_topic(TOPIC) or set()
    print(f"[monitor:{pod}] started…", flush=True)
    print(f"[monitor:{pod}] topic «{TOPIC}» partitions = {len(parts)}", flush=True)
    print(f"[monitor:{pod}] retrain cooldown = {RETRAIN_COOLDOWN_S}s", flush=True)

    # 启动资源采样（组件名固定为 'monitor'，并会写 host_resources.csv）
    stop_probe = start_probe("monitor")

    # 窗口里**同时保存** (x, y)，重训时才能写出 (X,y)
    window: Deque[Tuple[np.ndarray, Optional[float]]] = collections.deque(maxlen=DRIFT_WINDOW)
    baseline_x = None

    t_start = time.time()
    last_rx = time.time()
    all_sentinels = 0
    retrain_in_progress = False
    lock_started_ts: Optional[float] = None
    chosen_grid: Optional[str] = None
    cooldown_until: Optional[float] = None

    # 预热 baseline：收满 DRIFT_WINDOW
    while True:
        polled = consumer.poll(timeout_ms=1000)
        if not polled:
            if time.time() - t_start > MAX_WALL_SECS:
                print(f"[monitor:{pod}] max wall exceeded while warming baseline; exit.", flush=True)
                sync_all_metrics_to_minio(); stop_probe(); return
            continue
        for _, recs in polled.items():
            for rec in recs:
                last_rx = time.time()
                if is_sentinel(rec):
                    all_sentinels += 1
                    continue
                x, y = _xy_from_record(rec.value, feat_cols)
                window.append((x, y))
                if len(window) >= DRIFT_WINDOW:
                    baseline_x = np.stack([t[0] for t in window], axis=0)
                    print(f"[monitor:{pod}] baseline ready with {baseline_x.shape[0]} rows", flush=True)
                    break
        if baseline_x is not None:
            break

    stride_since_eval = 0

    while True:
        if (time.time() - last_rx) > MONITOR_IDLE_TIMEOUT_S:
            if retrain_in_progress and not _retrain_done_after(lock_started_ts or 0.0):
                time.sleep(1.0); continue
            print(f"[monitor:{pod}] idle > {MONITOR_IDLE_TIMEOUT_S}s; exit.", flush=True)
            break
        if time.time() - t_start > MAX_WALL_SECS:
            if retrain_in_progress and not _retrain_done_after(lock_started_ts or 0.0):
                time.sleep(1.0); continue
            print(f"[monitor:{pod}] max wall exceeded; exit.", flush=True)
            break

        polled = consumer.poll(timeout_ms=500)
        if not polled:
            time.sleep(0.05)
            # 重训完成检查
            if retrain_in_progress and _retrain_done_after(lock_started_ts or 0.0):
                dt = time.time() - (lock_started_ts or time.time())
                try: s3.delete_object(Bucket=BUCKET, Key=LOCK_KEY)
                except Exception: pass
                if MONITOR_SIGNAL_INFER_PAUSE:
                    try: s3.delete_object(Bucket=BUCKET, Key=PAUSE_INFER_KEY)
                    except Exception: pass
                retrain_in_progress = False
                cooldown_until = time.time() + max(0, RETRAIN_COOLDOWN_S)

                # 日志输出
                try:
                    raw = s3.get_object(Bucket=BUCKET, Key=DONE_KEY)["Body"].read()
                    info = json.loads(raw.decode("utf-8"))
                    extra = f"grid={info.get('grid')} rmse={info.get('best_rmse')} model={info.get('model_key')}"
                except Exception:
                    extra = f"grid={chosen_grid}"
                print(f"[monitor:{pod}] retrain done ({extra}) after {dt:.3f}s — baseline refreshed; cooldown {RETRAIN_COOLDOWN_S}s.", flush=True)
                log_metric(component="monitor", event="retrain_done_seen", grid=chosen_grid, wall_s=round(dt,3))

                if len(window) >= DRIFT_WINDOW:
                    baseline_x = np.stack([t[0] for t in list(window)[-DRIFT_WINDOW:]], axis=0)
                continue
            continue

        for _, recs in polled.items():
            for rec in recs:
                last_rx = time.time()
                if is_sentinel(rec):
                    all_sentinels += 1
                    if all_sentinels >= max(1, len(parts)):
                        if retrain_in_progress and not _retrain_done_after(lock_started_ts or 0.0):
                            time.sleep(1.0); continue
                        print(f"[monitor:{pod}] all sentinels seen; exit.", flush=True)
                        sync_all_metrics_to_minio(); stop_probe(); return
                    continue

                x, y = _xy_from_record(rec.value, feat_cols)
                window.append((x, y))
                stride_since_eval += 1

                if retrain_in_progress:
                    if _retrain_done_after(lock_started_ts or 0.0):
                        dt = time.time() - (lock_started_ts or time.time())
                        try: s3.delete_object(Bucket=BUCKET, Key=LOCK_KEY)
                        except Exception: pass
                        if MONITOR_SIGNAL_INFER_PAUSE:
                            try: s3.delete_object(Bucket=BUCKET, Key=PAUSE_INFER_KEY)
                            except Exception: pass
                        retrain_in_progress = False
                        cooldown_until = time.time() + max(0, RETRAIN_COOLDOWN_S)

                        try:
                            raw = s3.get_object(Bucket=BUCKET, Key=DONE_KEY)["Body"].read()
                            info = json.loads(raw.decode("utf-8"))
                            extra = f"grid={info.get('grid')} rmse={info.get('best_rmse')} model={info.get('model_key')}"
                        except Exception:
                            extra = f"grid={chosen_grid}"
                        print(f"[monitor:{pod}] retrain done ({extra}) after {dt:.3f}s — baseline refreshed; cooldown {RETRAIN_COOLDOWN_S}s.", flush=True)
                        log_metric(component="monitor", event="retrain_done_seen", grid=chosen_grid, wall_s=round(dt,3))
                        if len(window) >= DRIFT_WINDOW:
                            baseline_x = np.stack([t[0] for t in list(window)[-DRIFT_WINDOW:]], axis=0)
                    continue

                # 冷却中：跳过评估
                if cooldown_until is not None and time.time() < cooldown_until:
                    continue

                if (baseline_x is not None) and len(window) >= DRIFT_WINDOW and stride_since_eval >= EVAL_STRIDE:
                    stride_since_eval = 0
                    cur_x = np.stack([t[0] for t in list(window)[-DRIFT_WINDOW:]], axis=0)
                    js = _compute_js(baseline_x, cur_x, HIST_BINS)

                    # —— 把 JS 写入资源采样的 js 字段 —— #
                    update_extra(js=js)

                    grid = None
                    if js >= JS_THR_C:   grid = "C"
                    elif js >= JS_THR_B: grid = "B"
                    elif js >= JS_THR_A: grid = "A"

                    if grid:
                        chosen_grid = grid
                        print(f"[monitor:{pod}] drift triggered: js={js:.3f} → grid {grid}", flush=True)
                        log_metric(component="monitor", event="drift", js=round(js, 6), grid=grid)

                        xs = cur_x
                        ys = np.array([t[1] for t in list(window)[-DRIFT_WINDOW:]], dtype=np.float32)
                        mask = ~np.isnan(ys)
                        if mask.any():
                            _save_latest_batch_xy(xs[mask], ys[mask], feat_cols)
                        else:
                            save_bytes(BATCH_KEY, b"", "application/octet-stream")

                        _signal_retrain(grid)
                        if MONITOR_SIGNAL_INFER_PAUSE:
                            save_bytes(PAUSE_INFER_KEY, b"1\n", "text/plain")
                        retrain_in_progress = True
                        lock_started_ts = time.time()
                        continue

                    log_metric(component="monitor", event="js_tick", js=round(js, 6))

    sync_all_metrics_to_minio()
    stop_probe()
    print(f"[monitor:{pod}] metrics synced; bye.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
