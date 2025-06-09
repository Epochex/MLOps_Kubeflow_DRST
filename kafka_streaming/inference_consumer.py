#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• 批量消费 Kafka → 推理
• baseline_model + current_model 产生两组预测
• 记录延迟 / CPU / GPU / Cold-start / container_latency
• RTT_ms, throughput_s
• 最后把所有批次的 perf 写入 CSV 并上传到 MinIO
"""
import os
import sys
import io
import json
import time
import queue
import threading
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import psutil
from kafka import KafkaConsumer

from shared.metric_logger import log_metric
from shared.minio_helper  import s3, save_np, save_bytes, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from shared.profiler      import Timer

# -------- 兼容旧 Baseline --------
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
try:
    from ml.train_offline_full import MLPBaseline
except ModuleNotFoundError:
    from ml.train_offline import MLPBaseline

device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

# -------- 目录 & helper --------
os.makedirs(f"/mnt/pvc/{RESULT_DIR}/timing", exist_ok=True)
os.makedirs("/mnt/pvc/models",               exist_ok=True)

def _fetch(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{key}")["Body"].read()

def _bytes_to_model(raw: bytes) -> nn.Module:
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()
    mdl = MLPBaseline(len(FEATURE_COLS)).to(device)
    mdl.load_state_dict(obj)
    mdl.eval()
    return mdl

# -------- 载入 scaler --------
with open("/mnt/pvc/models/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load("/mnt/pvc/models/scaler.pkl")

# -------- baseline_model (永不热重载) --------
baseline_raw   = _fetch("model.pt")
baseline_model = _bytes_to_model(baseline_raw)

# -------- current_model（可热重载） --------
curr_raw = baseline_raw
model_sig = hashlib.md5(curr_raw).hexdigest()
t0_load = time.perf_counter()
with open("/mnt/pvc/models/model.pt", "wb") as f:
    f.write(curr_raw)
current_model = _bytes_to_model(curr_raw)
model_loading_ms = round((time.perf_counter() - t0_load) * 1000, 3)

def hot_reload():
    """若 MinIO 上的 model.pt 已更新，则重新加载并记录部署延迟"""
    global current_model, model_sig, model_loading_ms
    raw = _fetch("model.pt")
    sig = hashlib.md5(raw).hexdigest()
    if sig == model_sig:
        return
    # 1. 计算部署延迟
    try:
        ts_bytes = _fetch("last_update_utc.txt")
        push_ts  = datetime.fromisoformat(ts_bytes.decode().strip("Z"))
        deploy_delay = (datetime.utcnow() - push_ts).total_seconds()
    except Exception:
        deploy_delay = None

    # 2. 热重载
    t0 = time.perf_counter()
    current_model = _bytes_to_model(raw)
    model_sig     = sig
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)
    with open("/mnt/pvc/models/model.pt", "wb") as f:
        f.write(raw)
    print(f"[infer] 🔄 reloaded new model ({sig[:8]})  {model_loading_ms} ms")

    # 3. 上报
    log_metric(
        component="infer",
        event="hot_reload",
        model_loading_ms=model_loading_ms,
        deploy_delay_s=round(deploy_delay,3) if deploy_delay is not None else ""
    )

# -------- Kafka → 队列 --------
q = queue.Queue()
def _listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-infer",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    for m in cons:
        q.put(m.value)

threading.Thread(target=_listener, daemon=True).start()

def _take_batch():
    buf, start = [], time.time()
    while len(buf) < BATCH_SIZE and time.time() - start < CONSUME_IDLE_S:
        try:
            buf.append(q.get(timeout=1))
        except queue.Empty:
            pass
    return buf

# ---------------- 主循环 ----------------
perf = []  # 收集所有批次的 metrics
rows, pred_orig_all, pred_adj_all = [], [], []
seen_any, first_batch = False, True
start_no_data = time.time()
MAX_WAIT_NO_DATA = 180
container_start = time.perf_counter()

print("[infer] consumer started …")

while True:
    batch = _take_batch()
    if not batch:
        if not seen_any:
            if time.time() - start_no_data > MAX_WAIT_NO_DATA:
                print("[infer] no data for 3 min → exit")
                break
            continue
        else:
            print("[infer] idle → exit")
            break

    seen_any = True
    # 冷启动上报
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start",
                   cold_start_ms=round(cold_ms,3))
        first_batch = False

    # 计算 RTT
    recv_ts = datetime.utcnow()
    rtts = []
    for r in batch:
        ts = r.get("send_ts")
        if ts:
            try:
                st = datetime.fromisoformat(ts.strip("Z"))
                rtts.append((recv_ts - st).total_seconds()*1000)
            except:
                pass
    avg_rtt = round(sum(rtts)/len(rtts),3) if rtts else None

    # 构造特征矩阵
    X = np.array([[r["features"].get(c,0.) for c in FEATURE_COLS] for r in batch], np.float32)
    Xs = scaler.transform(X)

    # 推理
    cpu0 = proc.cpu_times()
    t_start = time.perf_counter()
    with Timer("Inference", "infer"):
        with torch.no_grad():
            preds_adj  = current_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
            preds_orig = baseline_model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
    cpu1 = proc.cpu_times()
    t_end = time.perf_counter()


    wall = t_end - t_start
    cpu_used = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
    cpu_pct  = round(100 * cpu_used / wall,2) if wall else 0.0
    gpu_pct  = (
        torch.cuda.memory_allocated() /
        torch.cuda.get_device_properties(0).total_memory * 100
        if device == "cuda" else 0.0
    )
    container_latency_ms = wall * 1000
    throughput_s = round(len(batch) / wall,3)

    # 记录到本地列表 perf
    entry = {
        "utc": datetime.utcnow().isoformat()+"Z",
        "batch_size": len(batch),
        "latency_ms": round(wall*1000,3),
        "cpu_pct": cpu_pct,
        "gpu_mem_pct": round(gpu_pct,2),
        "model_loading_ms": model_loading_ms,
        "container_latency_ms": round(container_latency_ms,3),
        "rtt_ms": avg_rtt or "",
        "throughput_s": throughput_s
    }
    perf.append(entry)

    print(f"[infer] batch {len(perf):03d} | {entry['latency_ms']}ms | "
          f"CPU {cpu_pct:.1f}% | RTT {avg_rtt}ms | TP {throughput_s}/s")

    # 写指标
    log_metric(
        component="infer",
        event="batch_metrics",
        **{k: v for k, v in entry.items() if k not in ("utc",)}
    )

    # 收集预测，用于后续保存
    for r, yo, ya in zip(batch, preds_orig, preds_adj):
        r["pred_orig"], r["pred"] = float(yo), float(ya)
    rows.extend(batch)
    pred_orig_all.extend(preds_orig)
    pred_adj_all.extend(preds_adj)

    # 检查并热重载模型
    hot_reload()

# ---------- 保存推理结果 ----------
if rows:
    df = pd.DataFrame(rows)
    if "label" not in df.columns:
        df["label"] = np.nan

    # 旧文件（兼容以前脚本）
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred.npy",
            df["pred"].astype(np.float32).values)
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_true.npy",
            df["label"].astype(np.float32).values)
    save_np(f"{RESULT_DIR}/inference_pred.npy", df["pred"].values)
    save_np(f"{RESULT_DIR}/inference_true.npy", df["label"].values)

    # 新文件：orig / adj
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred_orig.npy",
            np.array(pred_orig_all, np.float32))
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred_adj.npy",
            np.array(pred_adj_all,  np.float32))
    save_np(f"{RESULT_DIR}/inference_pred_orig.npy", np.array(pred_orig_all))
    save_np(f"{RESULT_DIR}/inference_pred_adj.npy",  np.array(pred_adj_all))

# ---------- 保存性能日志 ----------
perf_path = f"/mnt/pvc/{RESULT_DIR}/timing/infer_perf.json"
with open(perf_path, "w") as fp:
    json.dump(perf, fp, indent=2)

save_bytes(f"{RESULT_DIR}/timing/infer_perf.json",
           json.dumps(perf, indent=2).encode(), "application/json")

print(f"[infer] DONE. rows={len(rows)}, perf_samples={len(perf)}")

# 2) 保存 perf 到 CSV
df_perf = pd.DataFrame(perf)
csv_local = f"/mnt/pvc/{RESULT_DIR}/infer_metrics.csv"
df_perf.to_csv(csv_local, index=False)

# 上传到 MinIO
with open(csv_local, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/infer_metrics.csv", fp.read(), "text/csv")

print(f"[infer] saved metrics CSV → {csv_local}")
print(f"[infer] DONE. rows={len(rows)}, perf_samples={len(perf)}")