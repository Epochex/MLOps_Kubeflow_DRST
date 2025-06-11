#!/usr/bin/env python3
"""
kafka_streaming/monitor.py
────────────────────────────────────────────────────────────
• Drift Detection 批量消费 + JS 计算
• 缓存所有指标到本地，主循环结束后统一写出并上传到 MinIO
• 异步触发 dynamic_retrain，避免阻塞主线程
• 同步所有指标到 metrics_summary.csv
• 新增：
    - 完整 batch 才计算 drift
    - 打印日志，便于诊断
    - 在 trigger_retrain 中记录完整 RTT 并打印
"""
import os
import json
import queue
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
from kafka import KafkaConsumer
from scipy.spatial.distance import jensenshannon

from shared.config        import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    JS_TRIGGER_THRESH, BATCH_SIZE, CONSUME_IDLE_S,
    AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT,
    RESULT_DIR
)
from shared.features     import FEATURE_COLS
from shared.minio_helper import load_csv, save_bytes
from shared.metric_logger import log_metric, sync_all_metrics_to_minio

# ─── 全局变量 ──────────────────────────────────────────────────────
consumer = None                   # KafkaConsumer 对象
q = queue.Queue()                 # 内存队列
producer_done = threading.Event() # 收到 producer_done 标志后置位

# ─── 预加载基线数据 ─────────────────────────────────────────────────
print("[monitor] loading baseline data…")
baseline_df = load_csv(f"{DATA_DIR}/combined.csv")
print(f"[monitor] baseline loaded, {len(baseline_df)} rows")
CPU_FEATS   = [
    c for c in FEATURE_COLS
    if any(x in c for x in ("instructions","cycles","cache","branches"))
]

# ─── 本地缓存所有监测指标 ─────────────────────────────────────────────
metrics_buffer: list[dict] = []

def record_metric(component: str, event: str, **kwargs):
    """
    缓存一次打点，并同步到 metrics_summary.csv
    """
    entry = {
        "utc": datetime.utcnow().isoformat() + "Z",
        "component": component,
        "event": event,
        **kwargs
    }
    metrics_buffer.append(entry)
    log_metric(component=component, event=event, **kwargs)
    print(f"[monitor] metric recorded → {component}/{event} {kwargs}")

# ─── JS 计算函数 ──────────────────────────────────────────────────────
def js(a: np.ndarray, b: np.ndarray, bins: int=50, eps: float=1e-9) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    comb = np.concatenate([a, b])
    lo, hi = np.percentile(comb, [0.5, 99.5])
    if lo >= hi:
        return 0.0
    p, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    q_, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    p = p.astype(float) + eps; q_ = q_.astype(float) + eps
    p /= p.sum(); q_ /= q_.sum()
    val = jensenshannon(p, q_)**2
    return float(val) if not np.isnan(val) else 0.0

def avg_js(df: pd.DataFrame) -> float:
    feats = [f for f in CPU_FEATS if f in df.columns]
    vals = [js(baseline_df[f].values, df[f].values) for f in feats]
    return float(np.mean(vals)) if vals else 0.0

# ─── Kafka 监听线程 ─────────────────────────────────────────────────
def _listener():
    global consumer
    print("[monitor] starting Kafka listener…")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        group_id="cg-monitor",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    print(f"[monitor] subscribed to topic {KAFKA_TOPIC}")

    os.makedirs("/mnt/pvc/results", exist_ok=True)
    with open("/mnt/pvc/results/monitor_ready.flag", "w"):
        pass
    print("[monitor] wrote readiness flag")

    for msg in consumer:
        data = msg.value
        if data.get("producer_done"):
            print("[monitor] received producer_done signal")
            producer_done.set()
            continue
        q.put((msg.offset, data))
        print(f"[monitor] queued message offset={msg.offset}")

threading.Thread(target=_listener, daemon=True).start()

# ─── 系统 / Kafka 辅助 ───────────────────────────────────────────────
proc = psutil.Process()

def _sys_metrics() -> dict:
    times = psutil.cpu_times_percent(interval=None)
    return {
        "cpu_pct":     psutil.cpu_percent(interval=None),
        "mem_pct":     psutil.virtual_memory().percent,
        "io_wait_pct": getattr(times, "iowait", 0.0),
    }

def _take_batch():
    """
    尝试拉取最多 BATCH_SIZE 条消息，超时 CONSUME_IDLE_S 后返回当前已有的所有
    """
    buf, start = [], time.time()
    print(f"[monitor] taking batch (target {BATCH_SIZE})…")
    while len(buf) < BATCH_SIZE and time.time() - start < CONSUME_IDLE_S:
        try:
            buf.append(q.get(timeout=1))
        except queue.Empty:
            pass
    print(f"[monitor] batch ready, size={len(buf)}")
    return buf

# ─── 修改点：记录从检测到触发到重训脚本执行完成的端到端 RTT ─────────────────
def trigger_retrain(js_val: float):
    print(f"[monitor] retrain thread start (js_val={js_val:.4f})")
    t_start = time.time()
    # 同步调用 retrain，等到部署完成后才返回
    ret = os.system(f"python -m ml.dynamic_retrain {js_val:.4f}")
    t_end = time.time()
    rtt_ms = round((t_end - t_start) * 1000, 3)
    record_metric(
        component="monitor",
        event="retrain_rtt",
        js_val=round(js_val, 4),
        rtt_ms=rtt_ms
    )
    print(f"[monitor] retrain thread done (return={ret}), RTT={rtt_ms}ms")
# ────────────────────────────────────────────────────────────────────────────

# ─── 主循环 ─────────────────────────────────────────────────────────
print("[monitor] main loop start …")
last_retrain_ts = 0.0

while True:
    batch = _take_batch()
    # 全部消费者结束且队列空了，就退出
    if not batch:
        if producer_done.is_set() and q.empty():
            print("[monitor] producer_done & queue empty → exiting")
            break
        print("[monitor] no batch, continue…")
        continue

    offsets, rows = zip(*batch)
    # 仅当拿到完整 batch 大小时才算一次检测
    if len(batch) < BATCH_SIZE:
        print(f"[monitor] only got {len(batch)} (<{BATCH_SIZE}), skip drift detection")
        continue

    valid = [r for r in rows if "features" in r]
    if not valid:
        print("[monitor] batch has no features, skip")
        continue

    # ─── Drift Detection ────────────────────────────────────────────
    print(f"[monitor] running drift detection on batch_size={len(valid)}")
    t0 = time.time()
    df_sample = pd.DataFrame([r["features"] for r in valid])
    js_val = avg_js(df_sample)
    js_time_ms = (time.time() - t0) * 1000

    tp = next(iter(consumer.assignment())) if consumer and consumer.assignment() else None
    kafka_lag = max(0, consumer.end_offsets([tp])[tp] - offsets[-1] - 1) if tp else 0

    record_metric(
        component="monitor",
        event="drift_calc",
        batch_size=len(valid),
        js_val=round(js_val, 4),
        kafka_lag=kafka_lag,
        **_sys_metrics(),
        runtime_ms=round(js_time_ms, 3),
    )

    print(f"[monitor] drift={js_val:.4f}, runtime_ms={js_time_ms:.3f}ms, kafka_lag={kafka_lag}")

    # ─── 触发 retrain（阈值 + 最少 30s 冷却）──────────────────────────
    if js_val > JS_TRIGGER_THRESH and time.time() - last_retrain_ts >= 30:
        last_retrain_ts = time.time()
        np.save("/mnt/pvc/latest_batch.npy", valid)
        print(f"[monitor] drift {js_val:.4f} > threshold {JS_TRIGGER_THRESH} → spawning retrain")
        threading.Thread(target=trigger_retrain, args=(js_val,), daemon=True).start()

        record_metric(
            component="monitor",
            event="retrain_trigger",
            js_val=round(js_val, 4),
            update_trigger_delay_s=0.0,
        )
    else:
        print(f"[monitor] no retrain: js_val={js_val:.4f}, "
              f"cooldown={(time.time()-last_retrain_ts):.1f}s since last")

print("[monitor] main loop end")

# ─── 主循环结束后一次性写出并上传所有指标 ─────────────────────────────
os.makedirs("/mnt/pvc/results", exist_ok=True)
csv_path   = "/mnt/pvc/results/monitor_metrics.csv"
jsonl_path = "/mnt/pvc/results/monitor_metrics.jsonl"

pd.DataFrame(metrics_buffer).to_csv(csv_path, index=False)
with open(jsonl_path, "w") as fp:
    for entry in metrics_buffer:
        fp.write(json.dumps(entry) + "\n")

print("[monitor] saving CSV/JSONL → MinIO")
with open(csv_path, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.csv", fp.read(), "text/csv")
with open(jsonl_path, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.jsonl", fp.read(), "application/json")

sync_all_metrics_to_minio()
print("[monitor] uploaded all metrics to MinIO")

# 写 KFP V2 metadata.json
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json", "w") as f:
    json.dump({}, f)
print("[monitor] metadata.json written, exiting.")
