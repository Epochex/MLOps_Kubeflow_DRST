#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• 批量消费 Kafka → 推理（消费者组并行，每 pod 拉取自己分区）
• RTT 计算改为仅网络延迟（send_ts → 第一时刻 recv_ts）
• 模型热重载异步执行，避免阻塞主消费循环
"""
import os
import io
import json
import time
import queue
import threading
import hashlib
from datetime import datetime
from collections import deque

import numpy as np
import joblib
import psutil
import torch
import torch.nn as nn
from kafka import KafkaConsumer

from shared.metric_logger import log_metric
from shared.profiler      import Timer
from shared.minio_helper  import s3, save_bytes, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from ml.model             import DynamicMLP

# —— 本 Pod 的唯一标识 & 输出子目录 ——  
pod_name = os.getenv("HOSTNAME", "infer")
local_out = f"/mnt/pvc/{RESULT_DIR}/{pod_name}"
os.makedirs(local_out, exist_ok=True)

# —— 设备 & 进程统计 ——  
device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

# —— Helper: 从 MinIO 拉 artefact ——  
def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{key}")["Body"].read()

def _bytes_to_model(raw: bytes) -> nn.Module:
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()
    # 如果存的是 state_dict
    mdl = DynamicMLP(in_dim=len(FEATURE_COLS), hidden_layers=(128,64,32))
    mdl.load_state_dict(obj)
    return mdl.eval()

# —— 载入 scaler ——  
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

# —— baseline & current model ——  
baseline_raw   = _fetch("model.pt")
baseline_model = _bytes_to_model(baseline_raw)

curr_raw        = baseline_raw
model_sig       = hashlib.md5(curr_raw).hexdigest()
t0_load         = time.perf_counter()
with open(f"{local_out}/model.pt", "wb") as f:
    f.write(curr_raw)
current_model   = _bytes_to_model(curr_raw)
model_loading_ms = round((time.perf_counter() - t0_load) * 1000, 3)

# —— 异步热重载 ——  
def _reload_model():
    global current_model, model_sig, model_loading_ms
    raw = _fetch("model.pt"); sig = hashlib.md5(raw).hexdigest()
    if sig == model_sig:
        return
    # 计算部署延迟
    try:
        tsb = _fetch("last_update_utc.txt")
        push_ts = datetime.fromisoformat(tsb.decode().strip("Z"))
        deploy_delay = (datetime.utcnow() - push_ts).total_seconds()
    except:
        deploy_delay = None

    t0 = time.perf_counter()
    new_model = _bytes_to_model(raw)
    new_sig   = sig
    load_ms   = round((time.perf_counter() - t0)*1000, 3)

    # 替换模型
    current_model = new_model
    model_sig     = new_sig
    model_loading_ms = load_ms
    # 持久化到本地
    with open(f"{local_out}/model.pt", "wb") as f:
        f.write(raw)

    # 打点
    log_metric(
        component="infer",
        event="hot_reload",
        model_loading_ms=load_ms,
        deploy_delay_s=round(deploy_delay,3) if deploy_delay else ""
    )

def hot_reload():
    """在后台线程里执行热重载，主循环不阻塞。"""
    threading.Thread(target=_reload_model, daemon=True).start()

# —— Kafka 消费队列 & EOS 标志 ——  
q = queue.Queue()
producer_done = threading.Event()

def _listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-infer",               # <<-- 相同 group_id，Kafka 会分配分区并行消费
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )

    # 写就绪标志
    os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)
    flag_path = f"/mnt/pvc/{RESULT_DIR}/consumer_ready_{pod_name}.flag"
    with open(flag_path, "w"):
        pass
    print(f"[infer:{pod_name}] readiness flag → {flag_path}")

    for msg in cons:
        data = msg.value
        # —— 标记第一时刻接收时间 —— 
        data["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        if data.get("producer_done"):
            producer_done.set()
            continue
        q.put(data)

# 启动监听线程
threading.Thread(target=_listener, daemon=True).start()

# —— 批次拉取：首条阻塞，后续非阻塞 ——  
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

# —— 30s 一次的 Forecast Engine ——  
forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist:
            continue
        with Timer("Forecasting_Engine", "infer"):
            _ = float(np.mean(forecast_hist))
        log_metric(component="infer", event="forecast_once")

threading.Thread(target=_forecast_loop, daemon=True).start()

# ===== 主推理循环 =====  
perf, rows     = [], []
pred_o, pred_a = [], []
seen_any, first_batch = False, True
start_no_data = time.time()
MAX_WAIT      = 180
container_start = time.perf_counter()

print(f"[infer:{pod_name}] consumer started …")
while True:
    batch = _take_batch()
    # —— 一拿到 batch，就马上记录 recv_ts —— 
    recv = datetime.utcnow()

    # EOS & 空队列判断
    if not batch and producer_done.is_set():
        print(f"[infer:{pod_name}] received EOS & queue empty → exit")
        break
    if not batch and not seen_any:
        if time.time() - start_no_data > MAX_WAIT:
            print(f"[infer:{pod_name}] no data for 3 min → exit")
            break
        continue
    if not batch and seen_any:
        print(f"[infer:{pod_name}] idle → exit")
        break

    seen_any = True
    # 首次批次：打 cold start
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start", cold_start_ms=round(cold_ms,3))
        first_batch = False

    # —— Extraction —— 
    with Timer("Extraction", "infer"):
        rows_batch = list(batch)

    # —— Preprocessing —— 
    with Timer("Preprocessing", "infer"):
        X = np.array([[r["features"].get(c,0.) for c in FEATURE_COLS] for r in rows_batch], np.float32)
        Xs = scaler.transform(X)

    # —— Inference Engine —— 
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            pa = current_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
            po = baseline_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()

    wall = t1 - t0
    cpu_used = (cpu1.user+cpu1.system) - (cpu0.user+cpu0.system)
    cpu_pct  = round(100 * cpu_used / wall, 2) if wall else 0.0
    gpu_pct  = (torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory*100) if device=="cuda" else 0.0
    c_lat    = round(wall*1000,3)
    tp_s     = round(len(batch)/wall,3)

    # —— RTT 计算（send_ts → _recv_ts）—— 
    rtts = []
    for r in rows_batch:
        send_ts = r.get("send_ts"); recv_ts = r.get("_recv_ts")
        if send_ts and recv_ts:
            try:
                st = datetime.fromisoformat(send_ts.strip("Z"))
                rt = (datetime.fromisoformat(recv_ts.strip("Z")) - st).total_seconds()*1000
                rtts.append(rt)
            except:
                pass
    avg_rtt = round(sum(rtts)/len(rtts),3) if rtts else None

    # 收集指标
    entry = {
        "utc": datetime.utcnow().isoformat()+"Z",
        "component": "infer",
        "batch_size": len(batch),
        "latency_ms": round(wall*1000,3),
        "cpu_pct": cpu_pct,
        "gpu_mem_pct": round(gpu_pct,2),
        "model_loading_ms": model_loading_ms,
        "container_latency_ms": c_lat,
        "rtt_ms": avg_rtt or "",
        "throughput_s": tp_s
    }
    perf.append(entry)

    print(f"[infer:{pod_name}] batch {len(perf):03d} | {entry['latency_ms']} ms | CPU {cpu_pct}% | RTT {avg_rtt} ms | TP {tp_s}/s")
    log_metric(component="infer", event="batch_metrics", **{k:v for k,v in entry.items() if k not in ("utc","component")})

    # 保存预测值到 rows
    for r, yo, ya in zip(rows_batch, po, pa):
        r["pred_orig"], r["pred"] = float(yo), float(ya)
    rows.extend(rows_batch)
    pred_o.extend(po); pred_a.extend(pa)
    forecast_hist.extend(pa)

    # 检查并加载新模型（异步）
    hot_reload()

# ===== 结束后统一保存 & 上传所有结果 =====  
if rows:
    # 本地保存
    np.save(f"{local_out}/inference_pred.npy", np.array(pred_a,  np.float32))
    np.save(f"{local_out}/inference_true.npy", np.array([r.get("label",np.nan) for r in rows], np.float32))
    np.save(f"{local_out}/inference_pred_orig.npy", np.array(pred_o, np.float32))
    np.save(f"{local_out}/inference_pred_adj.npy", np.array(pred_a, np.float32))

    # 上传到 MinIO
    save_bytes(f"{RESULT_DIR}/{pod_name}/inference_pred.npy",    np.array(pred_a, np.float32).tobytes(),    "application/npy")
    save_bytes(f"{RESULT_DIR}/{pod_name}/inference_true.npy",   np.array([r.get("label",np.nan) for r in rows], np.float32).tobytes(), "application/npy")
    save_bytes(f"{RESULT_DIR}/{pod_name}/inference_pred_orig.npy", np.array(pred_o, np.float32).tobytes(),    "application/npy")
    save_bytes(f"{RESULT_DIR}/{pod_name}/inference_pred_adj.npy",  np.array(pred_a, np.float32).tobytes(),    "application/npy")

    # 保存并上传 perf CSV/JSONL
    import pandas as pd
    df_perf = pd.DataFrame(perf)
    csv_local = f"{local_out}/infer_metrics.csv"
    df_perf.to_csv(csv_local, index=False)
    with open(csv_local, "rb") as fp:
        save_bytes(f"{RESULT_DIR}/{pod_name}_infer_metrics.csv", fp.read(), "text/csv")
    log_json = json.dumps(perf, indent=2)
    save_bytes(f"{RESULT_DIR}/{pod_name}_infer_perf.json", log_json.encode(), "application/json")

print(f"[infer:{pod_name}] DONE. rows={len(rows)}, perf_samples={len(perf)}")

# —— 写 KFP V2 metadata ——  
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json","w") as f:
    json.dump({}, f)
