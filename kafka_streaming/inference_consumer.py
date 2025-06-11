#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• Extraction / Preprocessing / Inference_Engine 用 perf_counter 打点
• Forecasting_Engine 后台线程 30s 打点，使用同一个 current_model 对最后一条样本做一次预测
• 不再使用 shared.profiler.Timer，全部靠 time.perf_counter() 减少测量误差
• Print 阶段完成及耗时，可在 Kubeflow UI 日志中查看
• 支持 idle-timeout 自动退出，结束时拷贝 metrics_summary.csv 为 per-pod CSV 并同步到 MinIO
"""
import os
import io
import json
import time
import queue
import threading
import hashlib
from datetime import datetime

import numpy as np
import joblib
import psutil
import torch
import torch.nn as nn
from kafka import KafkaConsumer

from shared.metric_logger import log_metric, sync_all_metrics_to_minio
# 移除了 Timer 的引用，使用 perf_counter 进行测量
from shared.minio_helper import s3, save_bytes, BUCKET
from shared.config       import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features     import FEATURE_COLS
from ml.model            import DynamicMLP, build_model

done_flag = f"/mnt/pvc/{RESULT_DIR}/producer_done.flag"

# ─── Pod & 输出目录 ────────────────────────────────────────────────
pod_name  = os.getenv("HOSTNAME", "infer")
local_out = f"/mnt/pvc/{RESULT_DIR}/{pod_name}"
os.makedirs(local_out, exist_ok=True)

# ─── 设备 & 进程统计 ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

# ─── 从 MinIO 拉 artefact ─────────────────────────────────────────
def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{key}")["Body"].read()

def _bytes_to_model(raw: bytes) -> nn.Module:
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    # obj 是 state_dict，从 last_model_config.json 里读超参
    hparams = None
    try:
        cfg_raw = s3.get_object(
            Bucket=BUCKET,
            Key=f"{MODEL_DIR}/last_model_config.json"
        )["Body"].read()
        hparams = json.loads(cfg_raw.decode())
    except:
        pass

    if hparams:
        model = build_model(hparams, len(FEATURE_COLS))
    else:
        # ← 如果没有配置文件，则使用默认结构
        model = DynamicMLP(in_dim=len(FEATURE_COLS), hidden_layers=(128,64,32))
    model.load_state_dict(obj)
    return model.to(device).eval()


# ─── 加载 scaler & current_model ─────────────────────────────────
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

# 载入模型二进制并计算签名
raw = _fetch("model.pt")
model_sig = hashlib.md5(raw).hexdigest()

# 测量加载时间
t_load = time.perf_counter()
current_model = _bytes_to_model(raw)
load_ms = round((time.perf_counter() - t_load) * 1000, 3)

# 将模型写入本地，供 hot-reload 复用
with open(f"{local_out}/model.pt", "wb") as f:
    f.write(raw)

print(f"[infer:{pod_name}] Loaded model, load_ms={load_ms}ms")


# ─── 热重载 ───────────────────────────────────────────────────────
def _reload_model():
    global current_model, model_sig
    raw_new = _fetch("model.pt")
    sig_new = hashlib.md5(raw_new).hexdigest()
    if sig_new == model_sig:
        return

    # 计算部署延迟
    try:
        tsb = _fetch("last_update_utc.txt")
        push_ts = datetime.fromisoformat(tsb.decode().strip("Z"))
        deploy_delay = (datetime.utcnow() - push_ts).total_seconds()
    except:
        deploy_delay = None

    # 测量 reload 耗时
    t0 = time.perf_counter()
    new_model = _bytes_to_model(raw_new)
    reload_ms = round((time.perf_counter() - t0) * 1000, 3)

    current_model = new_model
    model_sig = sig_new

    with open(f"{local_out}/model.pt", "wb") as f:
        f.write(raw_new)

    log_metric(
        component="infer",
        event="hot_reload_runtime",
        model_loading_ms=reload_ms,
        deploy_delay_s=round(deploy_delay, 3) if deploy_delay else ""
    )
    print(f"[infer:{pod_name}] hot_reload → {reload_ms}ms, delay={deploy_delay}s")

def hot_reload():
    threading.Thread(target=_reload_model, daemon=True).start()


# ─── Kafka Listener ────────────────────────────────────────────────
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
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)
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


# ─── 批次拉取 ───────────────────────────────────────────────────────
def _take_batch():
    buf = []
    try:
        buf.append(q.get(timeout=CONSUME_IDLE_S))
    except queue.Empty:
        return []
    while len(buf) < BATCH_SIZE:
        try:
            buf.append(q.get_nowait())
        except queue.Empty:
            break
    return buf


# ─── 后台 Forecasting Engine ────────────────────────────────────
# 用于保存最后一条特征
last_features = None

def _forecast_loop():
    while True:
        time.sleep(30)
        if last_features is None:
            continue

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = current_model(torch.from_numpy(last_features).to(device)).cpu().item()
        latency = round((time.perf_counter() - t0) * 1000, 3)

        log_metric(
            component="infer",
            event="forecasting_prediction",
            forecast_value=round(pred, 3),
            latency_ms=latency
        )
        print(f"[infer:{pod_name}] forecast→ {pred:.3f}, latency={latency}ms")

threading.Thread(target=_forecast_loop, daemon=True).start()


# ─── 主循环 & 空闲超时退出 ─────────────────────────────────────────
first_batch     = True
container_start = time.perf_counter()
last_data_time  = time.time()
IDLE_TIMEOUT_S  = 180

print(f"[infer:{pod_name}] consumer started…")

while True:
    batch = _take_batch()
    now   = time.time()

    if not batch:
        if os.path.exists(done_flag):
            print(f"[infer:{pod_name}] detected producer_done.flag → exit")
            break
        if now - last_data_time > IDLE_TIMEOUT_S:
            print(f"[infer:{pod_name}] idle >{IDLE_TIMEOUT_S}s → exit")
            break
        continue

    last_data_time = now

    # —— Cold start 打点 ——  
    if first_batch:
        cold_ms = round((time.perf_counter() - container_start) * 1000, 3)
        log_metric(component="infer", event="cold_start", cold_start_ms=cold_ms)
        print(f"[infer:{pod_name}] cold_start → {cold_ms}ms")
        first_batch = False

    # —— Extraction & Preprocessing ——  
    rows_batch = list(batch)
    X = np.array([[r["features"].get(c, 0.0) for c in FEATURE_COLS]
                  for r in rows_batch], np.float32)
    Xs = scaler.transform(X)

    # 保留最后一条样本的特征用于 Forecasting
    last_features = Xs[-1:].copy()

    # —— Inference Engine ——  
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with torch.no_grad():
        preds = current_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()

    wall_ms  = (t1 - t0) * 1000
    cpu_used = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) * 1000
    cpu_pct  = round(cpu_used / wall_ms, 2) if wall_ms else 0.0
    tp_s     = round(len(rows_batch) / ((t1 - t0) or 1e-3), 3)

    # Pod RTT 计算
    rtts = []
    for r in rows_batch:
        if "send_ts" in r and "_recv_ts" in r:
            try:
                st   = datetime.fromisoformat(r["send_ts"].rstrip("Z"))
                recv = datetime.fromisoformat(r["_recv_ts"].rstrip("Z"))
                rtts.append((recv - st).total_seconds() * 1000)
            except:
                pass
    avg_rtt = round(sum(rtts) / len(rtts), 3) if rtts else 0.0

    log_metric(
        component="infer",
        event="batch_metrics",
        batch_size=len(rows_batch),
        latency_ms=round(wall_ms, 3),
        throughput_s=tp_s,
        cpu_pct=cpu_pct,
        gpu_mem_pct=0.0,
        model_loading_ms=load_ms,
        container_latency_ms=round(wall_ms, 3),
        rtt_ms=avg_rtt
    )
    print(f"[infer:{pod_name}] batch_metrics: latency={wall_ms:.3f}ms, rtt={avg_rtt}ms, tp={tp_s}/s")

    # —— 异步热重载检查 ——  
    hot_reload()

# ─── 退出前，同步并上传 per-pod CSV & 全量指标 ───────────────────────
global_csv = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
if os.path.exists(global_csv):
    with open(global_csv, "rb") as f:
        save_bytes(f"{RESULT_DIR}/{pod_name}_infer_metrics.csv",
                   f.read(), "text/csv")
    print(f"[infer:{pod_name}] uploaded {pod_name}_infer_metrics.csv")

sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced, exiting.")
