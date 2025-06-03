#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• 批量消费 Kafka → 推理
• 记录延迟 / CPU / GPU 显存 / 模型加载耗时
• 结果写入 PVC & MinIO
"""

import os, sys, io, json, time, queue, threading, hashlib, datetime, psutil
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from kafka import KafkaConsumer

from shared.metric_logger import log_metric
from shared.minio_helper  import s3, save_np, save_bytes, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS

# -------- 兼容旧 Baseline --------
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, ROOT)
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

def _md5(b): return hashlib.md5(b).hexdigest()

# -------- 载入 scaler --------
with open("/mnt/pvc/models/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load("/mnt/pvc/models/scaler.pkl")

# -------- 载入模型（计时） --------
def _bytes_to_model(raw: bytes) -> nn.Module:
    obj = torch.load(io.BytesIO(raw), map_location=device)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()
    mdl = MLPBaseline(len(FEATURE_COLS)).to(device)
    mdl.load_state_dict(obj); mdl.eval(); return mdl

_raw = _fetch("model.pt")
model_sig = _md5(_raw)
t0_load = time.perf_counter()
with open("/mnt/pvc/models/model.pt", "wb") as f: f.write(_raw)
model = _bytes_to_model(_raw)
model_loading_ms = round((time.perf_counter() - t0_load) * 1000, 3)

# -------- 热重载 --------
def hot_reload():
    global model, model_sig, model_loading_ms
    raw = _fetch("model.pt"); sig = _md5(raw)
    if sig == model_sig:
        return
    t0 = time.perf_counter()
    model = _bytes_to_model(raw); model_sig = sig
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)
    with open("/mnt/pvc/models/model.pt", "wb") as f:
        f.write(raw)
    print(f"[infer] 🔄 reloaded new model ({sig[:8]})  {model_loading_ms} ms")

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
rows, perf = [], []
seen_any   = False               # 是否已经拿到过数据
start_no_data = time.time()      # 完全没有数据的计时
MAX_WAIT_NO_DATA = 180           # 3 min 安全退出，防止 Pipeline 卡死

print("[infer] consumer started …")

while True:
    batch = _take_batch()

    # ---------- 退出条件 ----------
    if not batch:
        if not seen_any:
            if time.time() - start_no_data > MAX_WAIT_NO_DATA:
                print("[infer] no data for 3 min → exit"); break
            continue               # 继续等待首批数据
        else:
            # 已经处理过数据，再空闲 CONSUME_IDLE_S 秒就退出
            print("[infer] idle → exit"); break

    # ---------- 收到数据 ----------
    seen_any = True
    X = np.array([[r["features"].get(c, 0.) for c in FEATURE_COLS]
                  for r in batch], np.float32)
    Xs = scaler.transform(X)

    t0   = time.perf_counter()
    cpu0 = proc.cpu_times()
    with torch.no_grad():
        preds = model(torch.from_numpy(Xs).to(device)).cpu().numpy().ravel()
    cpu1 = proc.cpu_times()
    t1   = time.perf_counter()

    wall = t1 - t0
    cpu_used = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
    cpu_pct  = round(100 * cpu_used / wall, 2) if wall else 0.0
    gpu_pct  = (
        torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        if device == "cuda" and torch.cuda.is_available() else 0.0
    )

    perf.append({
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "latency_ms": round(wall * 1000, 3),
        "batch_size": len(batch),
        "cpu_pct": cpu_pct,
        "gpu_mem_pct": round(gpu_pct, 2),
        "model_loading_ms": model_loading_ms,
    })

    print(f"[infer] batch {len(rows)//BATCH_SIZE+1:03d} | "
          f"{wall*1000:.1f} ms | CPU {cpu_pct:.1f}%")

    log_metric(
        component="infer",
        batch_size=len(batch),
        latency_ms=perf[-1]["latency_ms"],
        cpu_pct=cpu_pct,
        gpu_mem_pct=perf[-1]["gpu_mem_pct"],
        model_loading_ms=model_loading_ms,
    )

    for r, y in zip(batch, preds):
        r["pred"] = float(y)
    rows.extend(batch)

    hot_reload()

# ---------- 保存推理结果 ----------
if rows:
    df = pd.DataFrame(rows)
    if "label" not in df.columns:
        df["label"] = np.nan

    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_pred.npy",
            df["pred"].astype(np.float32).values)
    np.save(f"/mnt/pvc/{RESULT_DIR}/inference_true.npy",
            df["label"].astype(np.float32).values)

    save_np(f"{RESULT_DIR}/inference_pred.npy", df["pred"].values)
    save_np(f"{RESULT_DIR}/inference_true.npy", df["label"].values)

# ---------- 保存性能日志 ----------
perf_path = f"/mnt/pvc/{RESULT_DIR}/timing/infer_perf.json"
with open(perf_path, "w") as fp:
    json.dump(perf, fp, indent=2)

save_bytes(f"{RESULT_DIR}/timing/infer_perf.json",
           json.dumps(perf, indent=2).encode(), "application/json")

print(f"[infer] DONE. rows={len(rows)}, perf_samples={len(perf)}")
