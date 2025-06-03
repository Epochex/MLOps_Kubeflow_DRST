#!/usr/bin/env python3
"""
kafka_streaming/monitor.py
────────────────────────────────────────────────────────────
• 每 BATCH_SIZE=10 条计算 JS-divergence
• 记录 CPU/内存/IO-wait/Kafka lag 等系统指标
• JS>0.2 时保存 latest_batch.npy 并触发动态重训
"""

import os, sys, json, queue, threading, time, random, pathlib, datetime
import psutil
import numpy as np, pandas as pd
from kafka import KafkaConsumer

from shared.metric_logger import log_metric

# -------- 项目内部 --------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    JS_DEFAULT_THRESH, BATCH_SIZE, CONSUME_IDLE_S,
    AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features import FEATURE_COLS
from shared.minio_helper import load_csv

# ---------- baseline ----------
baseline_df = load_csv(f"{DATA_DIR}/combined.csv")

CPU_FEATS = [
    c for c in FEATURE_COLS
    if any(x in c for x in ("instructions", "cycles", "cache", "branches"))
]

def js(a, b):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    p, _ = np.histogram(a, 30, (lo, hi), density=True)
    q, _ = np.histogram(b, 30, (lo, hi), density=True)
    kl_pq = np.nan_to_num((p * np.log((p + 1e-12) / (q + 1e-12))).sum())
    kl_qp = np.nan_to_num((q * np.log((q + 1e-12) / (p + 1e-12))).sum())
    return float(0.5 * (kl_pq + kl_qp))

def avg_js(df: pd.DataFrame) -> float:
    feats = [f for f in CPU_FEATS if f in df.columns]
    if not feats:
        return 0.0
    return float(np.mean([js(baseline_df[f].values, df[f].values) for f in feats]))

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
    _consumer_ready.set()

    for msg in cons:                           # ▸▸ 保留 Message 对象（含 offset）
        q.put(msg)

threading.Thread(target=_listener, daemon=True).start()
_consumer_ready.wait()

# ---------- 资源采样 ----------
proc = psutil.Process(os.getpid())

def _sys_metrics() -> dict:
    c = psutil.cpu_times_percent(interval=None)
    mem = psutil.virtual_memory()
    return {
        "cpu_pct"    : psutil.cpu_percent(interval=None),
        "mem_pct"    : mem.percent,
        "io_wait_pct": getattr(c, "iowait", 0.0),
    }

def _take_batch():
    """返回 [(msg.value, msg.offset), …]"""
    buf, start = [], time.time()
    while len(buf) < BATCH_SIZE and time.time() - start < CONSUME_IDLE_S:
        try:
            m = q.get(timeout=1)
            buf.append((m.value, m.offset))
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

    rows, offsets = zip(*batch)
    df_sample = pd.DataFrame(
        [r["features"] for r in rows if random.random() < 0.15]
    )
    js_val = avg_js(df_sample)

    # ---- Kafka lag ----
    tp = next(iter(consumer.assignment()))
    end_offset = consumer.end_offsets([tp])[tp]
    last_offset = offsets[-1]
    lag = max(0, end_offset - last_offset - 1)

    # ---- 日志 ----
    log_metric(
        component="monitor",
        event="drift_calc",
        batch_size=len(batch),
        js_val=round(js_val, 4),
        kafka_lag=lag,
        **_sys_metrics(),
    )

    # ---- 触发 retrain ----
    if js_val <= JS_DEFAULT_THRESH:
        continue
    if time.time() - last_retrain_ts < 120:
        print("[monitor] cooldown"); continue

    last_retrain_ts = time.time()
    pathlib.Path("/mnt/pvc").mkdir(exist_ok=True)
    np.save("/mnt/pvc/latest_batch.npy", rows)

    retrain_start = datetime.datetime.utcnow()
    print(f"[monitor] drift {js_val:.3f} → call dynamic_retrain")

    try:
        import subprocess
        subprocess.Popen(
            ["python", "-m", "ml.dynamic_retrain", f"{js_val:.4f}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except Exception as e:
        print("dynamic_retrain error:", e)
        continue

    delay = (datetime.datetime.utcnow() - retrain_start).total_seconds()
    log_metric(
        component="monitor",
        event="retrain_trigger",
        js_val=round(js_val, 4),
        update_trigger_delay_s=round(delay, 3),
    )

print("[monitor] end")
