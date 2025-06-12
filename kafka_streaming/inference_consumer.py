#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py – 真·流式推理
────────────────────────────────────────────────────────────
• Extraction / Preprocessing / Inference_Engine 用 Timer 打点
• 来一条 Kafka 消息立即推理并埋点
• Idle 时也 hot_reload；退出前强制 hot_reload(sync)
• 热更依据 MinIO models/* + last_model_config.json
• 打印详细阶段日志，方便在 Kubeflow UI 查看
"""

from __future__ import annotations
import os, io, json, time, queue, threading, hashlib
from datetime import datetime
from collections import deque

import numpy as np
import joblib
import psutil
import torch
import torch.nn as nn
from kafka import KafkaConsumer

from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler      import Timer
from shared.minio_helper  import s3, save_bytes, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from ml.model             import DynamicMLP, build_model

# ────────────────────────────────────────────────────────────
# 0. 基础环境与路径
# ────────────────────────────────────────────────────────────
done_flag = f"/mnt/pvc/{RESULT_DIR}/producer_done.flag"
pod_name  = os.getenv("HOSTNAME", "infer")
local_out = f"/mnt/pvc/{RESULT_DIR}/{pod_name}"
os.makedirs(local_out, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

# ────────────────────────────────────────────────────────────
# 1. MinIO 拉取工具
# ────────────────────────────────────────────────────────────
def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{key}")["Body"].read()

def _bytes_to_model(raw: bytes) -> nn.Module:
    """
    raw 既可能是整模型，也可能是 state_dict。
    若为 state_dict，则读取 last_model_config.json 动态建网络。
    """
    obj = torch.load(io.BytesIO(raw), map_location=device)

    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    # state_dict –> 需要网络结构
    try:
        cfg = json.loads(_fetch("last_model_config.json").decode())
    except Exception:
        cfg = None

    model = build_model(cfg, len(FEATURE_COLS)) if cfg else \
            DynamicMLP(len(FEATURE_COLS), (128, 64, 32))
    model.load_state_dict(obj)
    return model.eval()

# ────────────────────────────────────────────────────────────
# 2. 加载 scaler & baseline model
# ────────────────────────────────────────────────────────────
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

baseline_raw   = _fetch("model.pt")
baseline_model = _bytes_to_model(baseline_raw)

curr_raw          = baseline_raw
model_sig         = hashlib.md5(curr_raw).hexdigest()
t0_load           = time.perf_counter()
with open(f"{local_out}/model.pt", "wb") as f:
    f.write(curr_raw)
current_model     = _bytes_to_model(curr_raw)
model_loading_ms  = round((time.perf_counter() - t0_load) * 1000, 3)
print(f"[infer:{pod_name}] ✓ model loaded ({model_loading_ms} ms)")

# ────────────────────────────────────────────────────────────
# 3. Hot-reload 机制
# ────────────────────────────────────────────────────────────
def _reload_model(force: bool = False):
    """若远端模型变更则重新加载；force=True 无条件重载。"""
    global current_model, model_sig, model_loading_ms
    raw = _fetch("model.pt")
    sig = hashlib.md5(raw).hexdigest()

    if not force and sig == model_sig:
        return  # unchanged

    t0 = time.perf_counter()
    current_model = _bytes_to_model(raw)
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)
    model_sig = sig
    with open(f"{local_out}/model.pt", "wb") as f:
        f.write(raw)

    log_metric(component="infer", event="hot_reload_runtime",
               model_loading_ms=model_loading_ms)
    print(f"[infer:{pod_name}] 🔄 hot_reload → {model_loading_ms} ms (force={force})")

def hot_reload():
    """异步调用，避免阻塞主线程。"""
    threading.Thread(target=_reload_model, daemon=True).start()

# ────────────────────────────────────────────────────────────
# 4. Kafka 消费线程
# ────────────────────────────────────────────────────────────
q: queue.Queue = queue.Queue()
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
    # readiness flag
    flag = f"/mnt/pvc/{RESULT_DIR}/consumer_ready_{pod_name}.flag"
    with open(flag, "w"): pass
    print(f"[infer:{pod_name}] readiness flag → {flag}")

    for msg in cons:
        data = msg.value
        data["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        if data.get("producer_done"):
            producer_done.set()
            continue
        q.put(data)

threading.Thread(target=_listener, daemon=True).start()

# ────────────────────────────────────────────────────────────
# 5. Forecasting Engine（可选任务，此处简单均值）
# ────────────────────────────────────────────────────────────
forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist:
            continue
        with Timer("Forecasting_Engine", "infer"):
            _ = float(np.mean(forecast_hist))
        log_metric(component="infer", event="forecasting_runtime")
        print(f"[infer:{pod_name}] Forecasting_Engine run, hist_len={len(forecast_hist)}")

threading.Thread(target=_forecast_loop, daemon=True).start()

# ────────────────────────────────────────────────────────────
# 6. 主循环：逐条推理
# ────────────────────────────────────────────────────────────
IDLE_TIMEOUT_S  = 180
container_start = time.perf_counter()
first_msg       = True
last_data_ts    = time.time()

print(f"[infer:{pod_name}] streaming loop start …")

while True:
    # ----- 拉取 1 条 -----
    try:
        row = q.get(timeout=CONSUME_IDLE_S)
    except queue.Empty:
        hot_reload()  # 空闲也试试热更新
        now = time.time()

        # 退出判定 1：Producer 完成
        if producer_done.is_set() or os.path.exists(done_flag):
            print(f"[infer:{pod_name}] producer_done detected → final hot_reload & exit")
            _reload_model(force=True)
            break

        # 退出判定 2：空闲超时
        if now - last_data_ts > IDLE_TIMEOUT_S:
            print(f"[infer:{pod_name}] idle>{IDLE_TIMEOUT_S}s → final hot_reload & exit")
            _reload_model(force=True)
            break
        continue

    last_data_ts = time.time()

    # ----- Cold-start 打点 -----
    if first_msg:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start",
                   cold_start_ms=round(cold_ms, 3))
        print(f"[infer:{pod_name}] cold_start → {cold_ms:.2f} ms")
        first_msg = False

    # ==========  A) Extraction ==========
    with Timer("Extraction", "infer"):
        X_raw = np.array([[row["features"].get(c, 0.0) for c in FEATURE_COLS]],
                         dtype=np.float32)
    print(f"[infer:{pod_name}] Extraction done")

    # ==========  B) Preprocessing =======
    with Timer("Preprocessing", "infer"):
        X_scaled = scaler.transform(X_raw)
    print(f"[infer:{pod_name}] Preprocessing done")

    # ==========  C) Inference_Engine ====
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            pred_adj  = current_model(torch.from_numpy(X_scaled).to(device)).cpu().item()
            pred_orig = baseline_model(torch.from_numpy(X_scaled).to(device)).cpu().item()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()
    print(f"[infer:{pod_name}] Inference_Engine done")

    # ==========  D) 指标 & 日志 ==========
    wall_ms  = (t1 - t0) * 1000
    cpu_used = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) * 1000
    cpu_pct  = round(cpu_used / wall_ms, 2) if wall_ms else 0.0

    rtt_ms = 0.0
    if "send_ts" in row:
        try:
            st = datetime.fromisoformat(row["send_ts"].rstrip("Z"))
            rt = (datetime.fromisoformat(row["_recv_ts"].rstrip("Z")) - st).total_seconds() * 1000
            rtt_ms = round(rt, 3)
        except Exception:
            pass

    log_metric(
        component="infer",
        event="single_metrics",
        latency_ms=round(wall_ms, 3),
        cpu_pct=cpu_pct,
        rtt_ms=rtt_ms,
        model_loading_ms=model_loading_ms,
        gpu_mem_pct=0.0
    )
    print(f"[infer:{pod_name}] single_metrics: "
          f"lat={wall_ms:.1f} ms | rtt={rtt_ms:.1f} ms | cpu={cpu_pct}%")

    # 收集预测历史
    forecast_hist.append(pred_adj)

    # 异步热更新
    hot_reload()

# ────────────────────────────────────────────────────────────
# 7. 退出收尾
# ────────────────────────────────────────────────────────────
global_csv = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
if os.path.exists(global_csv):
    with open(global_csv, "rb") as f:
        save_bytes(f"{RESULT_DIR}/{pod_name}_infer_metrics.csv",
                   f.read(), "text/csv")
    print(f"[infer:{pod_name}] metrics CSV uploaded as {pod_name}_infer_metrics.csv")

sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced – bye.")
