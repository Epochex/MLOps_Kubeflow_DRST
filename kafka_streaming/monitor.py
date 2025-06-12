#!/usr/bin/env python3
"""
kafka_streaming/monitor.py  – 真·流式 Drift 监控 (Window-based)
─────────────────────────────────────────────────────────────
• 每收到 1 条 Kafka 消息立即放入滑动窗口
• 窗口长度达到 WINDOW_SIZE (默认 50) 时计算一次 Jensen–Shannon
• 超过 JS_TRIGGER_THRESH 即刻触发 dynamic_retrain
• 保留 record_metric() → 本地 CSV/JSONL → 统一上传 MinIO
"""

from __future__ import annotations
import os, json, queue, threading, time, sys
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
from kafka import KafkaConsumer
from scipy.spatial.distance import jensenshannon

from shared.config        import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    JS_TRIGGER_THRESH, CONSUME_IDLE_S,
    AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT, RESULT_DIR
)
from shared.features     import FEATURE_COLS
from shared.minio_helper import load_csv, save_bytes
from shared.metric_logger import log_metric, sync_all_metrics_to_minio

# ----------------------------------------------------------------------
# 0. 全局 & 初始化
# ----------------------------------------------------------------------
WINDOW_SIZE = int(os.getenv("JS_WINDOW_SIZE", "50"))      # 滑动窗口大小
MIN_RETRAIN_INTERVAL = 30                                 # 秒，防抖
metrics_buffer: list[dict] = []                           # 缓冲指标
producer_done = threading.Event()                         # 生产者结束标志
q: queue.Queue = queue.Queue()                            # Kafka → Monitor

# 预加载基线数据
baseline_df = load_csv(f"{DATA_DIR}/combined.csv")
JS_FEATS = [f for f in FEATURE_COLS if f in baseline_df.columns]

# ----------------------------------------------------------------------
# 1. 指标记录工具
# ----------------------------------------------------------------------
def record_metric(component: str, event: str, **kwargs):
    """
    写入本地 CSV/JSONL，并同步一份到内存。
    """
    entry = {
        "utc": datetime.utcnow().isoformat() + "Z",
        "component": component,
        "event": event,
        **kwargs
    }
    metrics_buffer.append(entry)
    log_metric(component=component, event=event, **kwargs)

# 系统资源
_proc = psutil.Process()
def _sys_metrics() -> dict:
    t = psutil.cpu_times_percent(interval=None)
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "mem_pct": psutil.virtual_memory().percent,
        "io_wait_pct": getattr(t, "iowait", 0.0),
    }

# ----------------------------------------------------------------------
# 2. Kafka 监听线程
# ----------------------------------------------------------------------
def _listener():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-monitor",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )

    os.makedirs("/mnt/pvc/results", exist_ok=True)
    with open("/mnt/pvc/results/monitor_ready.flag", "w"): pass
    print("[monitor] readiness flag touched")

    for msg in consumer:
        data = msg.value
        if data.get("producer_done"):
            producer_done.set()
            continue
        q.put(data)

threading.Thread(target=_listener, daemon=True).start()

# ----------------------------------------------------------------------
# 3. JS 计算函数
# ----------------------------------------------------------------------
def _js(p: np.ndarray, q_: np.ndarray, bins: int = 50, eps: float = 1e-9) -> float:
    p = p[~np.isnan(p)]; q_ = q_[~np.isnan(q_)]
    if p.size == 0 or q_.size == 0:
        return 0.0
    comb = np.concatenate([p, q_])
    lo, hi = np.percentile(comb, [0.5, 99.5])
    if lo >= hi:
        return 0.0
    hp, _ = np.histogram(p,  bins=bins, range=(lo, hi))
    hq, _ = np.histogram(q_, bins=bins, range=(lo, hi))
    hp = hp.astype(float) + eps
    hq = hq.astype(float) + eps
    hp /= hp.sum(); hq /= hq.sum()
    return float(jensenshannon(hp, hq) ** 2)

def _avg_js(df: pd.DataFrame) -> float:
    vals = [_js(baseline_df[f].values, df[f].values) for f in JS_FEATS]
    return float(np.mean(vals)) if vals else 0.0

# ----------------------------------------------------------------------
# 4. Retrain 触发
# ----------------------------------------------------------------------
def _trigger_retrain(js_val: float):
    cmd = f"python -m ml.dynamic_retrain {js_val:.4f}"
    print(f"[monitor] ➜ exec: {cmd}")
    os.system(cmd)

# ----------------------------------------------------------------------
# 5. 主循环：逐条流式
# ----------------------------------------------------------------------
print(f"[monitor] start, WINDOW_SIZE={WINDOW_SIZE}, JS_THRESH={JS_TRIGGER_THRESH}")
win_rows: list[dict] = []
last_retrain_ts = 0.0
msg_count = 0

while True:
    # ---------- 拉取 1 条 ----------
    try:
        item = q.get(timeout=CONSUME_IDLE_S)
    except queue.Empty:
        if producer_done.is_set():
            print("[monitor] producer_done & queue empty → exit main loop")
            break
        continue

    msg_count += 1
    win_rows.append(item)
    if len(win_rows) > WINDOW_SIZE:
        win_rows.pop(0)

    # 窗口未满就继续等
    if len(win_rows) < WINDOW_SIZE:
        continue

    # ---------- 计算 JS ----------
    t0 = time.time()
    df_window = pd.DataFrame([r["features"] for r in win_rows])
    js_val = _avg_js(df_window)
    js_time_ms = (time.time() - t0) * 1000

    record_metric(
        component="monitor",
        event="drift_calc",
        js_val=round(js_val, 4),
        window_size=WINDOW_SIZE,
        msg_since_start=msg_count,
        runtime_ms=round(js_time_ms, 3),
        **_sys_metrics()
    )

    print(f"[monitor] JS={js_val:.4f}  "
          f"(thr={JS_TRIGGER_THRESH})  "
          f"msgs={msg_count}")

    # ---------- 触发 Retrain ----------
    if js_val > JS_TRIGGER_THRESH and time.time() - last_retrain_ts >= MIN_RETRAIN_INTERVAL:
        last_retrain_ts = time.time()

        # 保存当前窗口数据供 dynamic_retrain 使用
        np.save("/mnt/pvc/latest_batch.npy", win_rows)

        print(f"[monitor] drift {js_val:.3f} > {JS_TRIGGER_THRESH}  "
              f"→ trigger retrain (window={WINDOW_SIZE})")
        threading.Thread(target=_trigger_retrain, args=(js_val,), daemon=True).start()

        record_metric(
            component="monitor",
            event="retrain_trigger",
            js_val=round(js_val, 4),
            update_trigger_delay_s=0.0
        )

# ----------------------------------------------------------------------
# 6. 写出所有指标 & 上传 MinIO
# ----------------------------------------------------------------------
print("[monitor] writing buffered metrics …")
os.makedirs("/mnt/pvc/results", exist_ok=True)
csv_path   = "/mnt/pvc/results/monitor_metrics.csv"
jsonl_path = "/mnt/pvc/results/monitor_metrics.jsonl"

pd.DataFrame(metrics_buffer).to_csv(csv_path, index=False)
with open(jsonl_path, "w") as fp:
    for entry in metrics_buffer:
        fp.write(json.dumps(entry) + "\n")

with open(csv_path, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.csv", fp.read(), "text/csv")
with open(jsonl_path, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.jsonl", fp.read(), "application/json")

# 同步所有组件的 summary.csv / *_metrics.csv
sync_all_metrics_to_minio()

# KFP V2 metadata.json（占位）
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json", "w") as f:
    json.dump({}, f)

print("[monitor] metrics uploaded – bye.")
