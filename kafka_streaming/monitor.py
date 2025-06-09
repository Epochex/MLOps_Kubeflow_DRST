#!/usr/bin/env python3
"""
kafka_streaming/monitor.py
────────────────────────────────────────────────────────────
• Drift Detection 批量消费 + JS 计算
• 每 BATCH_SIZE 做一次 Timer("Drift_Detection")
"""
import os, sys, json, queue, threading, time, pathlib
from datetime import datetime

import psutil, numpy as np, pandas as pd
from kafka import KafkaConsumer

# ← 补上这一行，否则 js() 中调用不了 jensenshannon
from scipy.spatial.distance import jensenshannon

from shared.metric_logger import log_metric
from shared.profiler      import Timer

# ─── 从 config 里导入阈值等 ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__) + "/..")
from shared.config import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    JS_TRIGGER_THRESH,  # ← 触发 retrain 的阈值
    BATCH_SIZE, CONSUME_IDLE_S,
    AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features     import FEATURE_COLS
from shared.minio_helper import load_csv

# baseline 数据只计算一次
baseline_df = load_csv(f"{DATA_DIR}/combined.csv")
CPU_FEATS   = [
    c for c in FEATURE_COLS
    if any(x in c for x in ("instructions","cycles","cache","branches"))
]

def js(a: np.ndarray, b: np.ndarray, bins: int=50, eps:float=1e-9) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size==0 or b.size==0: return 0.0
    comb = np.concatenate([a,b])
    lo, hi = np.percentile(comb, [0.5,99.5])
    if lo>=hi: return 0.0
    p,_ = np.histogram(a, bins=bins, range=(lo,hi), density=True)
    q,_ = np.histogram(b, bins=bins, range=(lo,hi), density=True)
    p=(p+eps)/(p+eps).sum(); q=(q+eps)/(q+eps).sum()
    # 这里就可以正常调用 jensenshannon 了
    return float(jensenshannon(p, q)**2)

def avg_js(df: pd.DataFrame) -> float:
    feats = [f for f in CPU_FEATS if f in df.columns]
    vals = [js(baseline_df[f].values, df[f].values) for f in feats]
    return float(np.mean(vals)) if vals else 0.0

# Kafka → 内存队列
q = queue.Queue()
_consumer_ready = threading.Event()

def _listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        group_id="cg-monitor",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    global consumer
    consumer = cons
    cons.poll(timeout_ms=0)
    _consumer_ready.set()
    for msg in cons:
        data = msg.value
        # 跳过结束标志
        if data.get("producer_done"):
            continue
        q.put(msg)

threading.Thread(target=_listener, daemon=True).start()
_consumer_ready.wait()

proc = psutil.Process()
def _sys_metrics():
    return {
        "cpu_pct":    psutil.cpu_percent(interval=None),
        "mem_pct":    psutil.virtual_memory().percent,
        "io_wait_pct": getattr(psutil.cpu_times_percent(interval=None), "iowait", 0.0),
    }

def _take_batch():
    buf, start = [], time.time()
    while len(buf) < BATCH_SIZE and time.time() - start < CONSUME_IDLE_S:
        try:
            buf.append(q.get(timeout=1))
        except queue.Empty:
            pass
    return buf

print("[monitor] start …")
last_retrain_ts = 0.0

while True:
    batch = _take_batch()
    if not batch:
        print("[monitor] idle → exit")
        break

    # 只保留带 features 的消息
    rows   = [m.value for m in batch if "features" in m.value]
    offsets= [m.offset for m in batch]
    if not rows:
        continue

    # —— Drift Detection 打点 ——  
    with Timer("Drift_Detection", "monitor"):
        df_sample = pd.DataFrame([r["features"] for r in rows])
        js_val    = avg_js(df_sample)

    # —— Kafka lag 打点 ——  
    assigned = consumer.assignment()
    kafka_lag = 0
    if assigned:
        tp = next(iter(assigned))
        kafka_lag = max(0, consumer.end_offsets([tp])[tp] - offsets[-1] - 1)

    log_metric(
        component="monitor",
        event="drift_calc",
        batch_size=len(rows),
        js_val=round(js_val,4),
        kafka_lag=kafka_lag,
        **_sys_metrics()
    )

    # —— 用统一阈值触发 retrain ——  
    if js_val > JS_TRIGGER_THRESH and time.time() - last_retrain_ts >= 120:
        last_retrain_ts = time.time()
        pathlib.Path("/mnt/pvc").mkdir(exist_ok=True)
        np.save("/mnt/pvc/latest_batch.npy", rows)

        print(f"[monitor] drift {js_val:.3f} → call dynamic_retrain")
        import subprocess
        subprocess.Popen(
            ["python", "-m", "ml.dynamic_retrain", f"{js_val:.4f}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        delay = (time.time() - last_retrain_ts)
        log_metric(
            component="monitor",
            event="retrain_trigger",
            js_val=round(js_val,4),
            update_trigger_delay_s=round(delay,3),
        )

print("[monitor] end")

# 写出 Kubeflow V2 必需的 metadata.json
import json, os
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json","w") as f:
    json.dump({}, f)
