#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• baseline  : offline 训练得到的 MLP（Scaler→PCA→N 维输入）
• adaptive  : 热更新模型（Scaler 后直接吃 full_dim 特征）
• 两路预测 + 真实值全量落盘，供 plot_final.py 直接拼 Phase-1/2/3
• 所有原有指标埋点 / 日志完全保留
"""

import os, io, json, time, queue, threading, hashlib
from datetime import datetime
from collections import deque
from typing import List

import numpy as np
import joblib
import psutil
import torch
import torch.nn as nn
from kafka import KafkaConsumer

from shared.utils import _fetch, _bytes_to_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler      import Timer
from shared.minio_helper  import s3, save_bytes, save_np, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from ml.model             import DynamicMLP


# ---------- 常量 ----------------------------------------------------------
done_flag = f"/mnt/pvc/{RESULT_DIR}/producer_done.flag"
pod_name  = os.getenv("HOSTNAME", "infer")
local_out = f"/mnt/pvc/{RESULT_DIR}/{pod_name}"
os.makedirs(local_out, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

pred_orig_hist: List[float] = []
pred_adj_hist : List[float] = []
true_hist     : List[float] = []


# ---------- scaler / PCA / baseline --------------------------------------
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

try:
    with open(f"{local_out}/pca.pkl", "wb") as f:
        f.write(_fetch("pca.pkl"))
    pca = joblib.load(f"{local_out}/pca.pkl"); use_pca = True
except Exception:
    pca, use_pca = None, False

baseline_raw   = _fetch("baseline_model.pt")
baseline_model = _bytes_to_model(baseline_raw)

baseline_in_dim= baseline_model.net[0].in_features

curr_raw       = _fetch("model.pt")
current_model  = _bytes_to_model(curr_raw)

model_sig      = hashlib.md5(curr_raw).hexdigest()
model_loading_ms = 0.0

# ---------- 热重载 --------------------------------------------------------
def _reload_model(force: bool = False):
    global current_model, curr_raw, model_sig, model_loading_ms
    raw = _fetch("model.pt")
    sig = hashlib.md5(raw).hexdigest()
    if not force and sig == model_sig:
        return
    t0 = time.perf_counter()
    current_model  = _bytes_to_model(raw)
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)
    curr_raw, model_sig = raw, sig
    log_metric(component="infer", event="hot_reload_runtime",
               model_loading_ms=model_loading_ms)
    print(f"[infer:{pod_name}] hot-reloaded, load={model_loading_ms} ms")

def hot_reload(): threading.Thread(target=_reload_model, daemon=True).start()

# ---------- Kafka 监听线程 -----------------------------------------------
q = queue.Queue()
producer_done = threading.Event()

def _listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-infer",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    flag = f"/mnt/pvc/{RESULT_DIR}/consumer_ready_{pod_name}.flag"
    open(flag, "w").close()
    print(f"[infer:{pod_name}] readiness flag → {flag}")

    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            producer_done.set(); continue
        v["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        q.put(v)

threading.Thread(target=_listener, daemon=True).start()

def _take_batch():
    buf = []
    try: buf.append(q.get(timeout=CONSUME_IDLE_S))
    except queue.Empty: return buf
    while len(buf) < BATCH_SIZE:
        try: buf.append(q.get_nowait())
        except queue.Empty: break
    return buf

# ---------- Forecasting (demo) -------------------------------------------
forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist: continue
        with Timer("Forecasting_Engine", "infer"): _ = float(np.mean(forecast_hist))
        log_metric(component="infer", event="forecasting_runtime")
threading.Thread(target=_forecast_loop, daemon=True).start()

# ---------- ★ 新增：对齐 adaptive 输入维度 -------------------------------
def _align_adaptive_input(X_scaled: np.ndarray, model: nn.Module) -> np.ndarray:
    """根据 model 的首层 in_features 动态裁剪 / padding X_scaled"""
    in_dim = model.net[0].in_features
    if in_dim == X_scaled.shape[1]:
        return X_scaled
    elif in_dim < X_scaled.shape[1]:
        return X_scaled[:, :in_dim]                         # 截断
    else:
        pad = np.zeros((X_scaled.shape[0], in_dim - X_scaled.shape[1]), dtype=np.float32)
        return np.concatenate([X_scaled, pad], axis=1)      # 右侧补 0

# ---------- 主循环 --------------------------------------------------------
first_batch     = True
container_start = time.perf_counter()
IDLE_TIMEOUT_S  = 180
last_data_time  = time.time()

print(f"[infer:{pod_name}] consumer started…")

while True:
    batch = _take_batch()
    now   = time.time()

    if not batch:
        hot_reload()
        if os.path.exists(done_flag) or (now - last_data_time) > IDLE_TIMEOUT_S:
            _reload_model(force=True); break
        time.sleep(0.3); continue

    last_data_time = now
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start",
                   cold_start_ms=round(cold_ms, 3))
        first_batch = False

    with Timer("Extraction", "infer"):
        rows_batch = list(batch)

    with Timer("Preprocessing", "infer"):
        X_raw = np.array([[r["features"].get(c, 0.0) for c in FEATURE_COLS]
                          for r in rows_batch], np.float32)
        X_scaled = scaler.transform(X_raw)
        X_base   = pca.transform(X_scaled) if use_pca else X_scaled[:, :baseline_in_dim]

        # ★★ 对 adaptive 分支做动态对齐
        X_adpt   = _align_adaptive_input(X_scaled, current_model)

    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            preds_base = baseline_model(
                torch.from_numpy(X_base).to(device)).cpu().numpy().ravel()
            preds_adpt = current_model(
                torch.from_numpy(X_adpt).to(device)).cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()

    # ---------- 后处理 & 指标输出（保持原逻辑，下方未改） ----------
    pred_orig_hist.extend(preds_base.tolist())
    pred_adj_hist .extend(preds_adpt.tolist())
    true_hist     .extend([r["label"] for r in rows_batch])

    rtts = []
    for r in rows_batch:
        if "send_ts" not in r: continue
        try:
            st = datetime.fromisoformat(r["send_ts"].rstrip("Z"))
            rt = (datetime.fromisoformat(r["_recv_ts"].rstrip("Z")) - st
                  ).total_seconds() * 1000
            rtts.append(rt)
        except Exception: pass
    avg_rtt = round(np.mean(rtts), 3) if rtts else 0.0
    wall_ms = (t1 - t0) * 1000
    cpu_used = ((cpu1.user + cpu1.system) -
                (cpu0.user + cpu0.system)) * 1000
    cpu_pct = round(cpu_used / wall_ms, 2) if wall_ms else 0.0
    tp_s    = round(len(rows_batch) / ((t1 - t0) or 1e-3), 3)

    log_metric(
        component="infer", event="batch_metrics",
        batch_size=len(rows_batch), latency_ms=round(wall_ms, 3),
        throughput_s=tp_s, cpu_pct=cpu_pct, gpu_mem_pct=0.0,
        model_loading_ms=model_loading_ms,
        container_latency_ms=round(wall_ms, 3), rtt_ms=avg_rtt
    )

    forecast_hist.extend(preds_adpt)
    hot_reload()

# ---------- 收尾（保持不变） ---------------------------------------------
arr_adj  = np.asarray(pred_adj_hist , np.float32)
arr_orig = np.asarray(pred_orig_hist, np.float32)
arr_true = np.asarray(true_hist    , np.float32)

np.save(f"{local_out}/inference_pred_adj.npy" , arr_adj)
np.save(f"{local_out}/inference_pred_orig.npy", arr_orig)
np.save(f"{local_out}/inference_true.npy"     , arr_true)

save_np(f"{RESULT_DIR}/{pod_name}_inference_pred_adj.npy" , arr_adj)
save_np(f"{RESULT_DIR}/{pod_name}_inference_pred_orig.npy", arr_orig)
save_np(f"{RESULT_DIR}/{pod_name}_inference_true.npy"     , arr_true)
print(f"[infer:{pod_name}] prediction arrays saved – total {len(arr_true)} samples")

global_csv = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
if os.path.exists(global_csv):
    with open(global_csv, "rb") as f:
        save_bytes(f"{RESULT_DIR}/{pod_name}_infer_metrics.csv",
                   f.read(), "text/csv")
    print(f"[infer:{pod_name}] uploaded {pod_name}_infer_metrics.csv")

sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced, exiting.")
