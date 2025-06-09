#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• 批量消费 Kafka → 推理
• 首条等待 CONSUME_IDLE_S，后续立即非阻塞拉满批次
• Extraction / Preprocessing / Inference Engine / Forecasting Engine
  全部用 Timer 打点
"""
import os, sys, io, json, time, queue, threading, hashlib
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
from shared.minio_helper  import s3, save_np, save_bytes, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from ml.model             import DynamicMLP

# —— 设备 & 进程统计 ——  
device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

# —— 准备目录 ——  
os.makedirs(f"/mnt/pvc/{RESULT_DIR}/timing", exist_ok=True)
os.makedirs("/mnt/pvc/models",               exist_ok=True)

# —— 从 MinIO 拉 artefact 帮手 ——  
def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{key}")["Body"].read()

def _bytes_to_model(raw: bytes) -> nn.Module:
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()
    mdl = DynamicMLP(in_dim=len(FEATURE_COLS), hidden_layers=(128,64,32))
    mdl.load_state_dict(obj)
    return mdl.eval()

# —— 载入 scaler ——  
with open("/mnt/pvc/models/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load("/mnt/pvc/models/scaler.pkl")

# —— baseline & current model ——  
baseline_raw   = _fetch("model.pt")
baseline_model = _bytes_to_model(baseline_raw)

curr_raw        = baseline_raw
model_sig       = hashlib.md5(curr_raw).hexdigest()
t0_load         = time.perf_counter()
with open("/mnt/pvc/models/model.pt", "wb") as f:
    f.write(curr_raw)
current_model   = _bytes_to_model(curr_raw)
model_loading_ms = round((time.perf_counter() - t0_load) * 1000, 3)

# —— 热重载 ——  
def hot_reload():
    global current_model, model_sig, model_loading_ms
    raw = _fetch("model.pt"); sig = hashlib.md5(raw).hexdigest()
    if sig == model_sig:
        return
    try:
        tsb = _fetch("last_update_utc.txt")
        push_ts = datetime.fromisoformat(tsb.decode().strip("Z"))
        deploy_delay = (datetime.utcnow() - push_ts).total_seconds()
    except:
        deploy_delay = None
    t0 = time.perf_counter()
    current_model = _bytes_to_model(raw)
    model_sig     = sig
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)
    with open("/mnt/pvc/models/model.pt", "wb") as f:
        f.write(raw)
    log_metric(
        component="infer",
        event="hot_reload",
        model_loading_ms=model_loading_ms,
        deploy_delay_s=round(deploy_delay, 3) if deploy_delay else ""
    )

# —— Kafka Listener & EOS 标志 ——  
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
    for msg in cons:
        data = msg.value
        if data.get("producer_done"):
            producer_done.set()
            continue

        # —— （可选）立即上报网络+Kafka 传输延迟 rtt_net ——  
        send_ts = data.get("send_ts")
        if send_ts:
            try:
                st0 = datetime.fromisoformat(send_ts.strip("Z"))
                rtt0 = (datetime.utcnow() - st0).total_seconds() * 1000
                log_metric(component="infer", event="rtt_net", rtt_ms=round(rtt0, 3))
            except:
                pass

        q.put(data)

threading.Thread(target=_listener, daemon=True).start()

# ===== 修改点: 首条等超时，后续立即拉满 =====
def _take_batch():
    buf = []
    # 阻塞拿第一条，最多等 CONSUME_IDLE_S 秒
    try:
        buf.append(q.get(timeout=CONSUME_IDLE_S))
    except queue.Empty:
        return buf
    # 拿到之后非阻塞地拉满
    while len(buf) < BATCH_SIZE:
        try:
            buf.append(q.get_nowait())
        except queue.Empty:
            break
    return buf
# ======================================

# —— Forecasting Engine（每 30s） ——  
from collections import deque
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

# —— 主推理循环 ——  
perf, rows = [], []
pred_o, pred_a = [], []
seen_any, first_batch = False, True
start_no_data = time.time()
MAX_WAIT = 180
container_start = time.perf_counter()

print("[infer] consumer started …")
while True:
    batch = _take_batch()

    if not batch and producer_done.is_set():
        print("[infer] received EOS & queue empty → exit")
        break
    if not batch and not seen_any:
        if time.time() - start_no_data > MAX_WAIT:
            print("[infer] no data for 3 min → exit")
            break
        continue
    if not batch and seen_any:
        print("[infer] idle → exit")
        break

    seen_any = True
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start", cold_start_ms=round(cold_ms, 3))
        first_batch = False

    # — Extraction —  
    with Timer("Extraction", "infer"):
        rows_batch = [r for r in batch]

    # — Preprocessing —  
    with Timer("Preprocessing", "infer"):
        import numpy as np
        X = np.array([[r["features"].get(c,0.) for c in FEATURE_COLS] for r in rows_batch], np.float32)
        Xs = scaler.transform(X)

    # — Inference Engine —  
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            pa = current_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
            po = baseline_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()

    wall = t1 - t0
    cpu_used = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
    cpu_pct  = round(100 * cpu_used / wall, 2) if wall else 0.0
    gpu_pct  = (torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory*100) if device=="cuda" else 0.0
    c_lat    = round(wall*1000,3)
    tp_s     = round(len(batch)/wall,3)

    # 批量 RTT  
    recv = datetime.utcnow()
    rtts = []
    for r in rows_batch:
        ts = r.get("send_ts")
        if ts:
            try:
                st = datetime.fromisoformat(ts.strip("Z"))
                rtts.append((recv - st).total_seconds()*1000)
            except:
                pass
    avg_rtt = round(sum(rtts)/len(rtts),3) if rtts else None

    entry = {
        "utc": datetime.utcnow().isoformat()+"Z",
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

    print(f"[infer] batch {len(perf):03d} | {entry['latency_ms']} ms | CPU {cpu_pct:.1f}% | RTT {avg_rtt} ms | TP {tp_s}/s")
    log_metric(component="infer", event="batch_metrics",
               **{k:v for k,v in entry.items() if k!="utc"})

    for r, yo, ya in zip(rows_batch, po, pa):
        r["pred_orig"], r["pred"] = float(yo), float(ya)
    rows.extend(rows_batch)
    pred_o.extend(po); pred_a.extend(pa)
    forecast_hist.extend(pa)
    hot_reload()

# （保存结果与写 KFP metadata 略）…



# —— 保存所有预测结果 ——  
if rows:
    df_all = pd.DataFrame(rows)
    if "label" not in df_all.columns:
        df_all["label"] = np.nan

    # 本地保存 .npy
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred.npy", df_all["pred"].astype(np.float32).values)
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_true.npy", df_all["label"].astype(np.float32).values)
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred_orig.npy", np.array(pred_o, np.float32))
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred_adj.npy",  np.array(pred_a, np.float32))

    # 上传到 MinIO
    save_np(f"{RESULT_DIR}/inference_pred.npy", df_all["pred"].values)
    save_np(f"{RESULT_DIR}/inference_true.npy", df_all["label"].values)
    save_np(f"{RESULT_DIR}/inference_pred_orig.npy", np.array(pred_o))
    save_np(f"{RESULT_DIR}/inference_pred_adj.npy", np.array(pred_a))

# —— 写出 performance JSON + 上传 ——  
perf_json = json.dumps(perf, indent=2)
perf_path = f"/mnt/pvc/{RESULT_DIR}/timing/infer_perf.json"
with open(perf_path, "w") as fp:
    fp.write(perf_json)
save_bytes(f"{RESULT_DIR}/timing/infer_perf.json", perf_json.encode(), "application/json")

# —— 写出 performance CSV + 上传 ——  
df_perf = pd.DataFrame(perf)
csv_local = f"/mnt/pvc/{RESULT_DIR}/infer_metrics.csv"
df_perf.to_csv(csv_local, index=False)
with open(csv_local, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/infer_metrics.csv", fp.read(), "text/csv")

print(f"[infer] DONE. rows={len(rows)}, perf_samples={len(perf)}")

# —— 写入 Kubeflow V2 metadata ——  
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json", "w") as f:
    json.dump({}, f)
