#!/usr/bin/env python3
"""
drst_inference/online/inference_consumer.py
Lightweight online inference Consumer:
- Kafka I/O, sentinel, partition stats → via drst_common.kafka_io
- latest model/metrics, scaler, selected_feats → via drst_common.artefacts
- readiness/KFP placeholder → via drst_common.runtime
- metrics logging/sync → via drst_common.metric_logger
Logic remains consistent with original design (two paths: baseline/adaptive, hot-reload guard thread).
"""
from __future__ import annotations
import os, io, json, time, queue, threading, hashlib
from datetime import datetime
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import joblib
import psutil

from drst_common.config import (
    KAFKA_TOPIC, BATCH_SIZE, CONSUME_IDLE_S,
    RESULT_DIR,
)
from drst_common.kafka_io import create_consumer, create_producer, partitions_count
from drst_common.artefacts import (
    read_latest, load_model_by_key, load_scaler, load_selected_feats, md5_bytes
)
from drst_common.utils import _bytes_to_model, calculate_accuracy_within_threshold
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.minio_helper import save_bytes

RETRAIN_TOPIC = os.getenv("RETRAIN_TOPIC", KAFKA_TOPIC + "_infer_count")
GAIN_THR_PP   = float(os.getenv("GAIN_THRESHOLD_PP", "0.01"))  # ≥x 个百分点才热更
RELOAD_INTERVAL_S = int(os.getenv("RELOAD_INTERVAL_S", "30"))
IDLE_TIMEOUT_S    = int(os.getenv("IDLE_TIMEOUT_S", "30"))

TMP_DIR  = "/tmp/infer"
os.makedirs(TMP_DIR, exist_ok=True)
pod_name = os.getenv("HOSTNAME", "infer")
local_out = os.path.join(TMP_DIR, pod_name)
os.makedirs(local_out, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

# load artefacts
SELECTED_FEATS = load_selected_feats()
scaler = load_scaler()

# baseline
baseline_model, base_raw = load_model_by_key("baseline_model.pt")
baseline_model = baseline_model.to(device)
baseline_in_dim = baseline_model.net[0].in_features

# adaptive
current_model, curr_raw = load_model_by_key("model.pt")
current_model = current_model.to(device)
current_model._val_acc15 = 0.0
model_sig = md5_bytes(curr_raw)
model_loading_ms = 0.0

model_lock = threading.Lock()

def _align_to_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    if in_dim == X.shape[1]: return X
    if in_dim < X.shape[1]:  return X[:, :in_dim]
    pad = np.zeros((X.shape[0], in_dim - X.shape[1]), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

# hot-reload deamon
def _maybe_reload(force: bool = False):
    global current_model, curr_raw, model_sig, model_loading_ms
    latest = read_latest()
    if latest is None:
        return
    model_key, metrics_key, metrics = latest
    try:
        raw_mdl = s3_get(f"{model_key}")
    except Exception:
        mdl, raw = load_model_by_key(model_key)
        raw_mdl = raw
        new_model = mdl.to(device)
    else:
        new_model = _bytes_to_model(raw_mdl).to(device)

    sig = md5_bytes(raw_mdl)
    if not force and sig == model_sig:
        return

    new_acc  = float(metrics.get("acc@0.15", 0.0))
    base_acc = float(metrics.get("baseline_acc@0.15", 0.0))
    gain_pp  = new_acc - base_acc
    if not force and gain_pp < GAIN_THR_PP:
        print(f"[infer:{pod_name}] Δ{gain_pp:+.3f} pp < {GAIN_THR_PP} → skip reload")
        return

    t0 = time.perf_counter()
    with model_lock:
        current_model = new_model.eval()
        current_model._val_acc15 = new_acc
        curr_raw, model_sig = raw_mdl, sig
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)

    log_metric(component="infer", event="hot_reload_runtime",
               model_loading_ms=model_loading_ms)
    print(f"[infer:{pod_name}] hot-reloaded ✓ baseline={base_acc:.2f}% → new={new_acc:.2f}% (Δ{gain_pp:+.3f} pp)")

def _reload_daemon():
    while True:
        time.sleep(RELOAD_INTERVAL_S)
        try:
            _maybe_reload()
        except Exception as e:
            print(f"[infer:{pod_name}] reload daemon error → {e}")

from drst_common.minio_helper import s3
from drst_common.config import BUCKET, MODEL_DIR
def s3_get(model_key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{model_key}")["Body"].read()

# ---------------- Kafka consumer/ready flag ----------------
q = queue.Queue()
producer_done = threading.Event()
sentinel_seen = 0
sentinel_lock = threading.Lock()
NUM_PARTITIONS = 0

def _listener():
    global NUM_PARTITIONS, sentinel_seen
    cons = create_consumer(KAFKA_TOPIC, group_id="cg-infer")
    time.sleep(1.0)
    NUM_PARTITIONS = partitions_count(cons, KAFKA_TOPIC)
    print(f"[infer:{pod_name}] topic «{KAFKA_TOPIC}» partitions = {NUM_PARTITIONS}")

    # readiness
    touch_ready("consumer", pod_name)

    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            with sentinel_lock:
                sentinel_seen += 1
                print(f"[infer:{pod_name}] got sentinel {sentinel_seen}/{NUM_PARTITIONS}")
            producer_done.set()
            continue
        v["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        q.put(v)

threading.Thread(target=_listener, daemon=True).start()
threading.Thread(target=_reload_daemon, daemon=True).start()

# ---------------- main circle ----------------
def _take_batch():
    buf = []
    try: buf.append(q.get(timeout=CONSUME_IDLE_S))
    except queue.Empty: return buf
    while len(buf) < BATCH_SIZE:
        try: buf.append(q.get_nowait())
        except queue.Empty: break
    return buf

pred_orig_hist: List[float] = []
pred_adj_hist : List[float] = []
true_hist     : List[float] = []
ts_hist       : List[str]   = []

forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist: continue
        log_metric(component="infer", event="forecasting_runtime")
threading.Thread(target=_forecast_loop, daemon=True).start()

first_batch     = True
container_start = time.perf_counter()
last_data_time  = time.time()
msg_total       = 0
correct_count   = 0
total_count     = 0

prod = create_producer()

print(f"[infer:{pod_name}] consumer started…")
while True:
    batch = _take_batch()
    now   = time.time()

    if not batch and (now - last_data_time) > IDLE_TIMEOUT_S:
        try: _maybe_reload(force=True)
        except Exception as e: print(f"[infer:{pod_name}] final reload error → {e}")
        print(f"[infer:{pod_name}] idle >{IDLE_TIMEOUT_S}s, graceful exit. processed={msg_total}")
        break

    with sentinel_lock:
        all_done = (sentinel_seen >= NUM_PARTITIONS) if NUM_PARTITIONS else False
    if not batch:
        if all_done and q.empty():
            try: _maybe_reload(force=True)
            except Exception as e: print(f"[infer:{pod_name}] final reload error → {e}")
            print(f"[infer:{pod_name}] all sentinels seen, exit. processed={msg_total}")
            break
        time.sleep(0.3)
        continue

    last_data_time = now
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start", cold_start_ms=round(cold_ms, 3))
        first_batch = False

    # ======== Preprocess ========
    rows_batch = list(batch)
    X_raw = np.array([[r["features"].get(c, 0.0) for c in SELECTED_FEATS] for r in rows_batch],
                     dtype=np.float32)
    X_scaled = scaler.transform(X_raw)

    with model_lock:
        model_ref = current_model

    X_base = _align_to_dim(X_scaled, baseline_in_dim)
    X_adpt = _align_to_dim(X_scaled, model_ref.net[0].in_features)

    # ======== Inference ========
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with torch.no_grad():
        preds_base = baseline_model(torch.from_numpy(X_base).to(device)).cpu().numpy().ravel()
        preds_adpt = model_ref(torch.from_numpy(X_adpt).to(device)).cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()

    labels = np.array([r["label"] for r in rows_batch], np.float32)
    errs   = np.abs(preds_adpt - labels) / np.maximum(labels, 1e-8)
    batch_correct = int((errs <= 0.2).sum())
    batch_total   = len(labels)
    correct_count += batch_correct
    total_count   += batch_total
    cum_acc = correct_count / total_count

    print(f"[infer:{pod_name}] accuracy@0.2 → batch {batch_correct}/{batch_total}, cum {cum_acc:.3f}")
    log_metric(component="infer", event="cumulative_accuracy",
               threshold=0.2,
               batch_correct=batch_correct, batch_total=batch_total,
               cumulative_correct=correct_count, cumulative_total=total_count,
               cumulative_accuracy=round(cum_acc, 3))

    pct = [50, 80, 90, 95, 99]
    err_base = np.abs(preds_base - labels) / np.maximum(labels, 1e-8)
    err_adpt = np.abs(preds_adpt - labels) / np.maximum(labels, 1e-8)
    base_q = np.percentile(err_base, pct).round(3)
    adpt_q = np.percentile(err_adpt, pct).round(3)
    log_metric(component="infer", event="err_dist",
               base_p50=float(base_q[0]), base_p80=float(base_q[1]),
               base_p90=float(base_q[2]), base_p95=float(base_q[3]),
               base_p99=float(base_q[4]),
               adpt_p50=float(adpt_q[0]), adpt_p80=float(adpt_q[1]),
               adpt_p90=float(adpt_q[2]), adpt_p95=float(adpt_q[3]),
               adpt_p99=float(adpt_q[4]))

    # batch
    wall_ms  = (t1 - t0) * 1000
    cpu_used = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) * 1000
    cpu_pct  = round(cpu_used / (wall_ms or 1e-9), 2)
    tp_s     = round(len(rows_batch) / (wall_ms or 1e-3), 3)
    # RTT statistics
    rtts = []
    for r in rows_batch:
        if "send_ts" in r:
            try:
                st = datetime.fromisoformat(r["send_ts"].rstrip("Z"))
                rt = (datetime.fromisoformat(r.get("_recv_ts","").rstrip("Z")) - st).total_seconds()*1000
                rtts.append(rt)
            except Exception:
                pass
    avg_rtt = round(float(np.mean(rtts)), 3) if rtts else 0.0
    log_metric(component="infer", event="batch_metrics",
               batch_size=len(rows_batch), latency_ms=round(wall_ms,3),
               throughput_s=tp_s, cpu_pct=cpu_pct, gpu_mem_pct=0.0,
               model_loading_ms=model_loading_ms, container_latency_ms=round(wall_ms,3),
               rtt_ms=avg_rtt)

    # total metrics
    msg_total += len(rows_batch)
    ts_hist.extend([r.get("send_ts","") for r in rows_batch])
    pred_orig_hist.extend(preds_base.tolist())
    pred_adj_hist .extend(preds_adpt.tolist())
    true_hist     .extend(labels.tolist())
    try:
        prod.send(RETRAIN_TOPIC, {"processed": batch_total})
    except Exception:
        pass
    forecast_hist.extend(preds_adpt)

arr_adj  = np.asarray(pred_adj_hist , np.float32)
arr_orig = np.asarray(pred_orig_hist, np.float32)
arr_true = np.asarray(true_hist     , np.float32)
arr_ts   = np.asarray([datetime.fromisoformat(t.rstrip("Z")).timestamp() if t else np.nan for t in ts_hist],
                      np.float64)

npz_local = os.path.join(local_out, "inference_trace.npz")
np.savez(npz_local, ts=arr_ts, pred_adj=arr_adj, pred_orig=arr_orig, true=arr_true)
with open(npz_local, "rb") as f:
    save_bytes(f"{RESULT_DIR}/{pod_name}_inference_trace.npz", f.read(), "application/octet-stream")
print(f"[infer:{pod_name}] trace npz saved – total {len(arr_true)} samples")

sync_all_metrics_to_minio()
write_kfp_metadata()
print(f"[infer:{pod_name}] all metrics synced, exiting.")
