#!/usr/bin/env python3
"""
kafka_streaming/monitor.py
────────────────────────────────────────────────────────────
• 每 BATCH_SIZE=10 计算 JS-divergence（剔除 NaN）
• 记录系统指标、Kafka lag
• JS > 阈值时保存 latest_batch.npy 并触发动态重训
• Timer("Drift_Detection") 统计 runtime 和 CPU-time
"""
import os
import sys
import json
import queue
import threading
import time
import pathlib
from datetime import datetime

import psutil
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from scipy.spatial.distance import jensenshannon

from shared.metric_logger import log_metric
from shared.profiler      import Timer

# —— 项目内部 ——  
sys.path.insert(0, os.path.dirname(__file__) + "/..")
from shared.config       import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    JS_DEFAULT_THRESH, BATCH_SIZE, CONSUME_IDLE_S,
    AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features     import FEATURE_COLS
from shared.minio_helper import load_csv

# ---------- 加载 baseline 数据 ----------
baseline_df = load_csv(f"{DATA_DIR}/combined.csv")
CPU_FEATS = [
    c for c in FEATURE_COLS
    if any(x in c for x in ("instructions", "cycles", "cache", "branches"))
]

# ---------- 改进版 JS 计算函数（剔除 NaN） ----------
def js(a: np.ndarray, b: np.ndarray, bins: int = 50, eps: float = 1e-9) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    combined = np.concatenate([a, b])
    lo, hi = np.percentile(combined, [0.5, 99.5])
    if lo >= hi:
        return 0.0
    p, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(jensenshannon(p, q) ** 2)

def avg_js(df: pd.DataFrame) -> float:
    feats = [f for f in CPU_FEATS if f in df.columns]
    vals = [js(baseline_df[f].values, df[f].values) for f in feats]
    return float(np.mean(vals)) if vals else 0.0

# ---------- Kafka → 内存队列 ----------
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
    # 触发一次 poll 以完成 assignment
    cons.poll(timeout_ms=0)
    _consumer_ready.set()
    for msg in cons:
        q.put(msg)

threading.Thread(target=_listener, daemon=True).start()
_consumer_ready.wait()

# ---------- 系统指标 ----------
proc = psutil.Process()
def _sys_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    io_wait = getattr(psutil.cpu_times_percent(interval=None), "iowait", 0.0)
    return {
        "cpu_pct": cpu,
        "mem_pct": mem,
        "io_wait_pct": io_wait,
    }

def _take_batch():
    buf, start = [], time.time()
    while len(buf) < BATCH_SIZE and time.time() - start < CONSUME_IDLE_S:
        try:
            buf.append(q.get(timeout=1))
        except queue.Empty:
            pass
    return buf

# ---------- 主循环 ----------
last_retrain_ts = 0.0
print("[monitor] start …")

while True:
    batch = _take_batch()
    if not batch:
        print("[monitor] idle → exit")
        break

    rows    = [m.value for m in batch]
    offsets = [m.offset for m in batch]

    # 1️⃣ 监控 JS & 计时
    with Timer("Drift_Detection", "monitor"):
        df_sample = pd.DataFrame([r["features"] for r in rows])
        js_val    = avg_js(df_sample)

    # 2️⃣ Kafka lag（单分区示例）
    assigned = consumer.assignment()
    if assigned:
        tp = next(iter(assigned))
        end_offset = consumer.end_offsets([tp])[tp]
        last_offset = offsets[-1]
        kafka_lag = max(0, end_offset - last_offset - 1)
    else:
        kafka_lag = 0

    # 3️⃣ 写 drift_calc 指标
    log_metric(
        component="monitor",
        event="drift_calc",
        batch_size=len(rows),
        js_val=round(js_val, 4),
        kafka_lag=kafka_lag,
        **_sys_metrics()
    )

    # 4️⃣ 触发 retrain（节流 120s）
    if js_val > JS_DEFAULT_THRESH and time.time() - last_retrain_ts >= 120:
        last_retrain_ts = time.time()

        pathlib.Path("/mnt/pvc").mkdir(exist_ok=True)
        np.save("/mnt/pvc/latest_batch.npy", rows)

        print(f"[monitor] drift {js_val:.3f} → call dynamic_retrain")
        retrain_start = datetime.utcnow()

        import subprocess
        subprocess.Popen(
            ["python", "-m", "ml.dynamic_retrain", f"{js_val:.4f}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        delay = (datetime.utcnow() - retrain_start).total_seconds()
        log_metric(
            component="monitor",
            event="retrain_trigger",
            js_val=round(js_val, 4),
            update_trigger_delay_s=round(delay, 3),
        )

print("[monitor] end")
