#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• Extraction/Preprocessing/Inference_Engine 用 Timer 打点
• Hot-reload 时根据 MinIO 上的 last_model_config.json 恢复网络层结构
• Forecasting_Engine 后台线程 30s 打点
• Print 阶段完成及耗时，可在 Kubeflow UI 日志中查看
• 支持 idle-timeout 自动退出，结束时拷贝 metrics_summary.csv 为 per-pod CSV
"""
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
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from ml.model             import DynamicMLP, build_model

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
        cfg_raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/last_model_config.json")["Body"].read()
        hparams = json.loads(cfg_raw.decode())
    except:
        pass

    if hparams:
        model = build_model(hparams, len(FEATURE_COLS))
    else:
        # ← 把这里改成你的 offline/train_offline.py 中的默认结构
        model = DynamicMLP(in_dim=len(FEATURE_COLS), hidden_layers=(128,64,32))
    model.load_state_dict(obj)
    return model.eval()


# ─── 加载 scaler & baseline model ─────────────────────────────────
with open(f"{local_out}/scaler.pkl","wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

baseline_raw   = _fetch("model.pt")
baseline_model = _bytes_to_model(baseline_raw)

curr_raw          = baseline_raw
model_sig         = hashlib.md5(curr_raw).hexdigest()
t0_load           = time.perf_counter()
with open(f"{local_out}/model.pt","wb") as f:
    f.write(curr_raw)
current_model     = _bytes_to_model(curr_raw)
model_loading_ms  = round((time.perf_counter() - t0_load)*1000,3)
print(f"[infer:{pod_name}] Loaded model, load_ms={model_loading_ms}ms")

# ─── 热重载 ───────────────────────────────────────────────────────
def _reload_model():
    global current_model, model_sig, model_loading_ms
    raw = _fetch("model.pt"); sig = hashlib.md5(raw).hexdigest()
    if sig == model_sig: return

    # 部署延迟
    try:
        tsb = _fetch("last_update_utc.txt")
        push_ts = datetime.fromisoformat(tsb.decode().strip("Z"))
        deploy_delay = (datetime.utcnow() - push_ts).total_seconds()
    except:
        deploy_delay = None

    tbegin = time.perf_counter()
    new_model = _bytes_to_model(raw)
    load_ms   = round((time.perf_counter()-tbegin)*1000,3)

    current_model    = new_model
    model_sig        = sig
    model_loading_ms = load_ms
    with open(f"{local_out}/model.pt","wb") as f:
        f.write(raw)

    log_metric(
        component="infer",
        event="hot_reload_runtime",
        model_loading_ms=load_ms,
        deploy_delay_s=round(deploy_delay,3) if deploy_delay else ""
    )
    print(f"[infer:{pod_name}] hot_reload → {load_ms}ms, delay={deploy_delay}s")

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
    with open(flag,"w"): pass
    print(f"[infer:{pod_name}] readiness flag → {flag}")

    for msg in cons:
        data = msg.value
        data["_recv_ts"] = datetime.utcnow().isoformat()+"Z"
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
        return buf
    while len(buf) < BATCH_SIZE:
        try:
            buf.append(q.get_nowait())
        except queue.Empty:
            break
    return buf

# ─── 后台 Forecasting Engine ───────────────────────────────────────
forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist: continue
        with Timer("Forecasting_Engine", "infer"):
            _ = float(np.mean(forecast_hist))
        log_metric(component="infer", event="forecasting_runtime")
        print(f"[infer:{pod_name}] Forecasting_Engine run, hist_len={len(forecast_hist)}")

threading.Thread(target=_forecast_loop, daemon=True).start()

# ─── 主循环 & 空闲超时退出 ───────────────────────────────────────────
first_batch     = True
container_start = time.perf_counter()
IDLE_TIMEOUT_S  = 180
last_data_time  = time.time()
print(f"[infer:{pod_name}] consumer started…")

while True:
    batch = _take_batch()
    now   = time.time()

    # 如果拿到数据就重置 last_data_time
    if batch:
        last_data_time = now
    else:
        # 检测 Producer 写的全局结束标志文件
        if os.path.exists(done_flag):
            print(f"[infer:{pod_name}] detected producer_done.flag → exit")
            break
        # 空闲超时也退出
        if now - last_data_time > IDLE_TIMEOUT_S:
            print(f"[infer:{pod_name}] idle >{IDLE_TIMEOUT_S}s → exit")
            break
        continue  # 继续等待数据

    # —— Cold start 打点 ——  
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start",
                   cold_start_ms=round(cold_ms, 3))
        print(f"[infer:{pod_name}] cold_start → {cold_ms:.3f}ms")
        first_batch = False

    # —— Extraction 阶段 ——  
    with Timer("Extraction", "infer"):
        rows_batch = list(batch)
    print(f"[infer:{pod_name}] Extraction done: {len(rows_batch)} samples")

    # —— Preprocessing 阶段 ——  
    with Timer("Preprocessing", "infer"):
        X  = np.array([[r["features"].get(c, 0.0) for c in FEATURE_COLS]
                       for r in rows_batch], np.float32)
        Xs = scaler.transform(X)
    print(f"[infer:{pod_name}] Preprocessing done")

    # —— Inference Engine 阶段 ——  
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            preds_adj  = current_model(torch.from_numpy(Xs).to(device)) \
                           .cpu().numpy().ravel()
            preds_base = baseline_model(torch.from_numpy(Xs).to(device)) \
                           .cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()
    print(f"[infer:{pod_name}] Inference_Engine done")

    # —— RTT & Throughput 计算 ——  
    rtts = []
    for r in rows_batch:
        if "send_ts" in r and "_recv_ts" in r:
            try:
                st = datetime.fromisoformat(r["send_ts"].rstrip("Z"))
                rt = (datetime.fromisoformat(r["_recv_ts"].rstrip("Z"))
                      - st).total_seconds() * 1000
                rtts.append(rt)
            except:
                pass
    avg_rtt = round(sum(rtts) / len(rtts), 3) if rtts else 0.0
    wall_ms  = (t1 - t0) * 1000
    cpu_used = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) * 1000
    cpu_pct  = round(cpu_used / wall_ms, 2) if wall_ms else 0.0
    tp_s     = round(len(rows_batch) / ((t1 - t0) or 1e-3), 3)

    log_metric(
        component="infer",
        event="batch_metrics",
        batch_size=len(rows_batch),
        latency_ms=round(wall_ms, 3),
        throughput_s=tp_s,
        cpu_pct=cpu_pct,
        gpu_mem_pct=0.0,
        model_loading_ms=model_loading_ms,
        container_latency_ms=round(wall_ms, 3),
        rtt_ms=avg_rtt
    )
    print(f"[infer:{pod_name}] batch_metrics: "
          f"latency={wall_ms:.3f}ms, rtt={avg_rtt}ms, tp={tp_s}/s")

    # —— 为 Forecasting Engine 收集历史 ——  
    forecast_hist.extend(preds_adj)

    # —— 异步热重载检查 ——  
    hot_reload()

# ─── 退出前，把全局 metrics_summary.csv 拷贝为 per-pod CSV ─────────────
global_csv = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
if os.path.exists(global_csv):
    with open(global_csv, "rb") as f:
        save_bytes(f"{RESULT_DIR}/{pod_name}_infer_metrics.csv",
                   f.read(), "text/csv")
    print(f"[infer:{pod_name}] uploaded {pod_name}_infer_metrics.csv")

# ─── 同步所有日志指标到 MinIO ─────────────────────────────────────────
sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced, exiting.")
