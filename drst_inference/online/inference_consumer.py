# ===== 文件: drst_inference/online/inference_consumer.py =====
#!/usr/bin/env python3
from __future__ import annotations
import os, io, json, time, hashlib, threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

from drst_common import config as _cfg
from drst_common.config import (
    KAFKA_TOPIC, BATCH_SIZE, CONSUME_IDLE_S as _CFG_IDLE_S,
    MODEL_DIR, RESULT_DIR, BUCKET, INFER_STDOUT_EVERY, RELOAD_INTERVAL_S,
    GAIN_THR_PP, RETRAIN_TOPIC, TARGET_COL,
    INFER_RESPECT_PAUSE_FLAG, PAUSE_INFER_KEY
)
from drst_common.kafka_io import (
    create_consumer, create_producer, partitions_for_topic,
    SENTINEL_HEADER_KEY, SENTINEL_HEADER_VAL, SENTINEL_FALLBACK_VALUE
)
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.minio_helper import save_bytes, s3
from drst_common.artefacts import load_selected_feats, load_scaler, read_latest, load_model_by_key
from drst_common.resource_probe import start as start_probe

RUN_TAG  = os.getenv("KFP_RUN_ID") or os.getenv("PIPELINE_RUN_ID") or "drst"
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "cg-infer")

POD_NAME = os.getenv("HOSTNAME", "infer")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

try:
    IDLE_S = int(os.getenv("IDLE_TIMEOUT_S", "")) if os.getenv("IDLE_TIMEOUT_S") else int(_CFG_IDLE_S)
except Exception:
    IDLE_S = 60

ACC_THR  = float(getattr(_cfg, "ACC_THR", 0.25))
_thr_str = ("%.2f" % ACC_THR).rstrip("0").rstrip(".")

def _align_to_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    d = X.shape[1]
    if d == in_dim: return X
    if d > in_dim:  return X[:, :in_dim]
    pad = np.zeros((X.shape[0], in_dim - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

def _load_json_key(key: str) -> Dict:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return json.loads(raw.decode())
    except Exception:
        return {}

def _pick_metric(metrics: Dict[str, float]) -> Tuple[float, float]:
    candidates = [f"acc@{_thr_str}", "acc@0.15", "acc"]
    base_candidates = [f"baseline_acc@{_thr_str}", "baseline_acc@0.15", "baseline_acc"]
    new_acc = next((float(metrics[k]) for k in candidates if k in metrics), 0.0)
    base_acc = next((float(metrics[k]) for k in base_candidates if k in metrics), 0.0)
    return new_acc, base_acc

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
    return False

def _rows_to_matrix(rows: List[dict], feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    X = np.zeros((len(rows), len(feat_cols)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.float32)
    ts_list: List[Optional[str]] = []
    for i, r in enumerate(rows):
        feats = r.get("features", {})
        for j, c in enumerate(feat_cols):
            X[i, j] = float(feats.get(c, 0.0))
        y[i] = float(r.get("label", 0.0))
        ts_list.append(r.get("send_ts"))
    return X, y, ts_list

print(f"[infer:{POD_NAME}] loading artefacts…")
SELECTED_FEATS = load_selected_feats()
SCALER = load_scaler()

stop_probe = start_probe(f"infer_{POD_NAME}")

baseline_model, baseline_raw = load_model_by_key("baseline_model.pt")
baseline_model = baseline_model.to(DEVICE).eval()
baseline_in_dim = baseline_model.net[0].in_features
print(f"[infer:{POD_NAME}] baseline in_dim = {baseline_in_dim}")

_latest = read_latest()
_cur_key = (_latest[0] if _latest else "model.pt")
try:
    current_model, curr_raw = load_model_by_key(_cur_key)
    current_model = current_model.to(DEVICE).eval()
    print(f"[infer:{POD_NAME}] current model = {_cur_key}")
except Exception as e:
    print(f"[infer:{POD_NAME}] WARN load {_cur_key} failed: {e}; fallback -> baseline")
    current_model, curr_raw = baseline_model, baseline_raw
cur_sig = hashlib.md5(curr_raw).hexdigest()

_model_lock = torch.multiprocessing.get_context("spawn").RLock()
_model_loading_ms = 0.0

def _maybe_reload_model():
    global current_model, curr_raw, cur_sig, _model_loading_ms
    try:
        latest = read_latest()
        if not latest: return
        model_key, metrics_key, _ts = latest
        m, raw = load_model_by_key(model_key)
        sig = hashlib.md5(raw).hexdigest()
        if sig == cur_sig: return
        metrics = _load_json_key(metrics_key)
        new_acc, base_acc = _pick_metric(metrics)
        gain_pp  = new_acc - base_acc
        if ("acc@0.15" in metrics or f"acc@{_thr_str}" in metrics) and (gain_pp < GAIN_THR_PP):
            print(f"[infer:{POD_NAME}] skip reload: Δ{gain_pp:+.3f}pp < {GAIN_THR_PP}")
            return
        t0 = time.perf_counter()
        mdl = m.to(DEVICE).eval()
        _model_loading_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        with _model_lock:
            current_model, curr_raw, cur_sig = mdl, raw, sig
        log_metric(component="infer", event="hot_reload",
                   model_key=model_key, metrics_key=metrics_key,
                   gain_pp=round(gain_pp, 4), load_ms=_model_loading_ms)
        print(f"[infer:{POD_NAME}] hot reloaded -> {model_key} (Δacc@{_thr_str}={gain_pp:+.3f}pp, load={_model_loading_ms}ms)")
    except Exception as e:
        print(f"[infer:{POD_NAME}] reload error: {e}")

def _reload_daemon():
    while True:
        time.sleep(RELOAD_INTERVAL_S)
        _maybe_reload_model()
threading.Thread(target=_reload_daemon, daemon=True).start()

print(f"[infer:{POD_NAME}] connecting Kafka…")
consumer = create_consumer(KAFKA_TOPIC, group_id=GROUP_ID)
time.sleep(1.0)
num_parts = len(partitions_for_topic(KAFKA_TOPIC))
print(f"[infer:{POD_NAME}] topic «{KAFKA_TOPIC}» partitions = {num_parts}")

save_bytes(f"{RESULT_DIR}/consumer_ready_{POD_NAME}.flag", b"", "text/plain")
producer = create_producer()
sentinel_seen = 0
last_data_ts = time.time()

ts_hist:   List[float] = []
true_hist: List[float] = []
pred_b_hist: List[float] = []
pred_c_hist: List[float] = []

cum_correct = 0
cum_total   = 0
_batch_idx  = 0

paused = False

def _process_batch(rows: List[dict]):
    global cum_correct, cum_total, _batch_idx
    X_raw, labels, send_ts = _rows_to_matrix(rows, SELECTED_FEATS)
    X_sc = SCALER.transform(X_raw)

    with _model_lock:
        mdl = current_model
        in_dim_cur = mdl.net[0].in_features

    X_b = _align_to_dim(X_sc, baseline_in_dim)
    X_c = _align_to_dim(X_sc, in_dim_cur)

    t0 = time.perf_counter()
    with torch.no_grad():
        pb = baseline_model(torch.from_numpy(X_b).to(DEVICE)).cpu().numpy().ravel()
        pc = mdl            (torch.from_numpy(X_c).to(DEVICE)).cpu().numpy().ravel()
    wall_ms = (time.perf_counter() - t0) * 1000.0

    denom = np.maximum(np.abs(labels), 1e-8)
    err_c = np.abs(pc - labels) / denom
    batch_correct = int((err_c <= ACC_THR).sum())
    batch_total   = len(rows)
    cum_correct  += batch_correct
    cum_total    += batch_total
    cum_acc = (cum_correct / max(1, cum_total))

    err_b = np.abs(pb - labels) / denom
    pct   = [50, 80, 90, 95, 99]
    q_b   = np.percentile(err_b, pct).round(3)
    q_c   = np.percentile(err_c, pct).round(3)

    log_metric(
        component="infer", event="batch_metrics",
        batch_size=batch_total, latency_ms=round(wall_ms, 3),
        **{f"cumulative_accuracy@{_thr_str}": round(cum_acc, 3)},
        base_p50=float(q_b[0]), base_p80=float(q_b[1]), base_p90=float(q_b[2]),
        base_p95=float(q_b[3]), base_p99=float(q_b[4]),
        adpt_p50=float(q_c[0]), adpt_p80=float(q_c[1]), adpt_p90=float(q_c[2]),
        adpt_p95=float(q_c[3]), adpt_p99=float(q_c[4]),
        model_loading_ms=_model_loading_ms
    )

    _batch_idx += 1
    if INFER_STDOUT_EVERY > 0 and (_batch_idx % INFER_STDOUT_EVERY == 0):
        print(
            f"[infer:{POD_NAME}] batch#{_batch_idx} n={batch_total} t={wall_ms:.2f}ms "
            f"cum@{_thr_str}={cum_acc:.3f} "
            f"adpt[p50,p90,p95,p99]={q_c[0]:.3f},{q_c[2]:.3f},{q_c[3]:.3f},{q_c[4]:.3f} "
            f"base[p50,p90,p95,p99]={q_b[0]:.3f},{q_b[2]:.3f},{q_b[3]:.3f},{q_b[4]:.3f}",
            flush=True
        )

    pred_b_hist.extend(pb.tolist()); pred_c_hist.extend(pc.tolist()); true_hist.extend(labels.tolist())
    for ts in send_ts:
        try: t = datetime.fromisoformat(str(ts).rstrip("Z")).timestamp()
        except Exception: t = time.time()
        ts_hist.append(float(t))
    try:
        producer.send(RETRAIN_TOPIC, {"processed": batch_total})
    except Exception:
        pass

print(f"[infer:{POD_NAME}] ready; consuming…")
batch_buf: List[dict] = []
BATCH = max(1, BATCH_SIZE)

def _pause_flag_exists() -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=PAUSE_INFER_KEY)
        return True
    except Exception:
        return False

try:
    while True:
        # —— 可选暂停 —— #
        if INFER_RESPECT_PAUSE_FLAG and _pause_flag_exists():
            if not paused:
                # 确保拿到 assignment
                if not consumer.assignment():
                    consumer.poll(timeout_ms=100)
                parts = list(consumer.assignment())
                if parts:
                    consumer.pause(*parts)
                paused = True
                print(f"[infer:{POD_NAME}] paused by flag; waiting retrain to finish…", flush=True)
            last_data_ts = time.time()  # 避免因 idle 退出
            time.sleep(0.5)
            continue
        elif paused:
            parts = list(consumer.assignment())
            if parts:
                consumer.resume(*parts)
            paused = False
            print(f"[infer:{POD_NAME}] resumed; continue consuming.", flush=True)

        polled = consumer.poll(timeout_ms=1000)
        got_data = False
        for _, records in polled.items():
            for msg in records:
                if _is_sentinel_msg(msg):
                    sentinel_seen += 1
                    print(f"[infer:{POD_NAME}] got sentinel {sentinel_seen}/{num_parts}")
                    continue
                try:
                    v = json.loads(msg.value)
                    if isinstance(v, dict) and v.get("producer_done"):
                        sentinel_seen += 1
                        print(f"[infer:{POD_NAME}] got sentinel {sentinel_seen}/{num_parts}")
                        continue
                except Exception:
                    continue

                got_data = True
                last_data_ts = time.time()
                batch_buf.append(v)
                if len(batch_buf) >= BATCH:
                    _process_batch(batch_buf); batch_buf.clear()

        if not got_data:
            if batch_buf:
                _process_batch(batch_buf); batch_buf.clear()
            if num_parts and sentinel_seen >= num_parts:
                print(f"[infer:{POD_NAME}] all sentinels received; exiting loop"); break
            if (time.time() - last_data_ts) > IDLE_S:
                print(f"[infer:{POD_NAME}] idle > {IDLE_S}s; exiting loop"); break
finally:
    arr_adj  = np.asarray(pred_c_hist,  np.float32)
    arr_orig = np.asarray(pred_b_hist,  np.float32)
    arr_true = np.asarray(true_hist,    np.float32)
    arr_ts   = np.asarray(ts_hist,      np.float64)
    local_npz = f"/tmp/{POD_NAME}_inference_trace.npz"
    np.savez(local_npz, ts=arr_ts, pred_adj=arr_adj, pred_orig=arr_orig, true=arr_true)
    with open(local_npz, "rb") as f:
        save_bytes(f"{RESULT_DIR}/{POD_NAME}_inference_trace.npz", f.read(), "application/octet-stream")
    print(f"[infer:{POD_NAME}] trace saved: {len(arr_true)} samples")

    sync_all_metrics_to_minio()
    stop_probe()
    print(f"[infer:{POD_NAME}] done.")
