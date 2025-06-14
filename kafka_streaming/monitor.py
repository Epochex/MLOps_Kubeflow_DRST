#!/usr/bin/env python3
"""
kafka_streaming/monitor.py ─ 流式 Drift 监控
──────────────────────────────────────────────────────────────
• Kafka 消息 ⇒ 滑动窗口 ⇒ Jensen–Shannon
• 超阈值则后台触发 dynamic_retrain，并把窗口写入 /mnt/pvc/latest_batch.npy
"""
from __future__ import annotations
import os, json, queue, threading, time, subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
from kafka import KafkaConsumer
from scipy.spatial.distance import jensenshannon

from shared.config import (
    KAFKA_TOPIC, KAFKA_SERVERS,
    JS_TRIGGER_THRESH, JS_SEV1_THRESH, JS_SEV2_THRESH,
    CONSUME_IDLE_S, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT,
    RESULT_DIR
)
from shared.features      import FEATURE_COLS
from shared.minio_helper  import load_csv, save_bytes
from shared.metric_logger import log_metric, sync_all_metrics_to_minio

# ----------------------------------------------------------------------
# 0. 全局 & 初始化
# ----------------------------------------------------------------------
WINDOW_SIZE          = int(os.getenv("JS_WINDOW_SIZE", "50"))   # JS 计算窗口
RETRAIN_BATCH_SIZE   = int(os.getenv("RETRAIN_BATCH_SIZE", "500"))   # 喂给最新模型retrain 的样本数
MIN_RETRAIN_INTERVAL = int(os.getenv("MIN_RETRAIN_INTERVAL", "300"))  # 5 min

metrics_buffer : list[dict] = []
producer_done  = threading.Event()
q: queue.Queue = queue.Queue()

from collections import deque
retrain_buf = deque(maxlen=RETRAIN_BATCH_SIZE)   

retrain_lock   = threading.Lock()
retrain_running = False

# —— 预加载 baseline（离线训练分布）——
BASELINE_KEY = os.getenv("BASELINE_KEY", "datasets_old/old_total.csv")
baseline_df  = load_csv(BASELINE_KEY)
baseline_df.drop(columns=["input_rate", "latency", "output_rate"],
                 errors="ignore", inplace=True)
baseline_df  = baseline_df.reindex(columns=FEATURE_COLS, fill_value=0.0)
JS_FEATS     = [f for f in FEATURE_COLS if f in baseline_df.columns]

# ----------------------------------------------------------------------
# 1. 指标工具
# ----------------------------------------------------------------------
def record_metric(component: str, event: str, **kw):
    entry = {
        "utc": datetime.utcnow().isoformat() + "Z",
        "component": component,
        "event": event,
        **kw
    }
    metrics_buffer.append(entry)
    log_metric(component=component, event=event, **kw)

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
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-monitor",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    os.makedirs("/mnt/pvc/results", exist_ok=True)
    open("/mnt/pvc/results/monitor_ready.flag", "w").close()
    print("[monitor] readiness flag touched")
    # —— 以下为新增，同步一个空文件到 MinIO，Producer 可在 MinIO 上轮询
    from shared.minio_helper import save_bytes
    save_bytes(f"{RESULT_DIR}/monitor_ready.flag", b"", "text/plain")

    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            producer_done.set(); continue
        q.put(v)

threading.Thread(target=_listener, daemon=True).start()

# ----------------------------------------------------------------------
# 3. Jensen–Shannon
# ----------------------------------------------------------------------
def _js(p: np.ndarray, q_: np.ndarray, bins=50, eps=1e-9) -> float:
    p = p[~np.isnan(p)]; q_ = q_[~np.isnan(q_)]
    if p.size == 0 or q_.size == 0: return 0.0
    comb = np.concatenate([p, q_])
    lo, hi = np.percentile(comb, [0.5, 99.5])
    if lo >= hi: return 0.0
    hp,_ = np.histogram(p , bins=bins, range=(lo,hi))
    hq,_ = np.histogram(q_, bins=bins, range=(lo,hi))
    hp = hp.astype(float)+eps; hq = hq.astype(float)+eps
    hp/=hp.sum(); hq/=hq.sum()
    return float(jensenshannon(hp,hq))

def _avg_js(df: pd.DataFrame) -> float:
    return float(np.mean([_js(baseline_df[f].values, df[f].values)
                          for f in JS_FEATS])) if JS_FEATS else 0.0

# ----------------------------------------------------------------------
# 4. 重训后台线程
# ----------------------------------------------------------------------
def _bg_retrain(js_val: float, snapshot_rows: list[dict]):
    """
    ① 把当前窗口写入 latest_batch.npy（供 dynamic_retrain）
    ② 调 dynamic_retrain（同步等待）
    ③ 扩充 baseline_df
    """
    global retrain_running, baseline_df
    try:
        start_ts = time.time()
        print(f"[monitor] retrain start at {datetime.utcnow().isoformat()}Z (JS={js_val:.4f})")

        # 1) Dump latest window for retrain process
        np.save("/mnt/pvc/latest_batch.npy", np.array(snapshot_rows, dtype=object))
        print("[monitor] latest_batch.npy saved")

        # 2) 同步调用 retrain 子进程
        with retrain_lock:
            retrain_running = True
        cmd = ["python", "-m", "ml.dynamic_retrain", f"{js_val:.4f}"]
        print(f"[monitor] ➜ exec: {' '.join(cmd)}")
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[monitor] retrain exit code = {ret}")

        # 3) 更新 baseline（线程安全）
        with retrain_lock:
            df_add = pd.DataFrame([r["features"] for r in snapshot_rows])
            baseline_df = pd.concat([baseline_df, df_add], ignore_index=True)
            retrain_running = False

        elapsed = time.time() - start_ts
        print(f"[monitor] retrain done at {datetime.utcnow().isoformat()}Z – elapsed {elapsed:.2f}s & baseline updated")

    except Exception as exc:
        print(f"[monitor] retrain thread error: {exc}")
        with retrain_lock:
            retrain_running = False

# ----------------------------------------------------------------------
# 5. 主循环
# ----------------------------------------------------------------------
print(f"[monitor] start, WINDOW={WINDOW_SIZE}, THR={JS_TRIGGER_THRESH}")
win_rows: list[dict] = []
last_retrain_ts = 0.0
msg_count = 0

while True:
    try:
        item = q.get(timeout=CONSUME_IDLE_S)
    except queue.Empty:
        if producer_done.is_set():
            print("[monitor] producer_done & queue empty → bye")
            break
        continue

    msg_count += 1
    win_rows.append(item)  # js滑窗
    retrain_buf.append(item)  # 用于 dynamic_retrain 缓存
    
    if len(win_rows) > WINDOW_SIZE:
        win_rows.pop(0)
    if len(win_rows) < WINDOW_SIZE:
        continue

    t0 = time.time()
    df_win = pd.DataFrame([r["features"] for r in win_rows])
    js_val = _avg_js(df_win)
    js_ms  = (time.time() - t0) * 1000

    record_metric("monitor", "drift_calc",
                  js_val=round(js_val, 4), window_size=WINDOW_SIZE,
                  msg_since_start=msg_count, runtime_ms=round(js_ms, 3),
                  **_sys_metrics())

    print(f"[monitor] JS={js_val:.4f} (thr={JS_TRIGGER_THRESH}) msgs={msg_count}")

    # ---- 触发重训 ----
    now = time.time()
    if (js_val > JS_TRIGGER_THRESH
        and (now - last_retrain_ts) >= MIN_RETRAIN_INTERVAL
        and not retrain_running):
        last_retrain_ts = now
        snapshot = list(retrain_buf)

        # severity
        if js_val > JS_SEV2_THRESH:
            severity = "K"
        elif js_val > JS_SEV1_THRESH:
            severity = "2"
        else:
            severity = "1"

        # print(f"[monitor] Begin:retrain triggered at msg={msg_count} → Severity-{severity} (JS={js_val:.4f})")

        threading.Thread(target=_bg_retrain,
                         args=(js_val, snapshot),
                         daemon=True).start()

        record_metric("monitor", "retrain_trigger",
                      js_val=round(js_val, 4), severity=severity)

# ----------------------------------------------------------------------
# 6. 收尾：写指标 & 上传
# ----------------------------------------------------------------------
print("[monitor] writing buffered metrics …")
os.makedirs("/mnt/pvc/results", exist_ok=True)
csv_path   = "/mnt/pvc/results/monitor_metrics.csv"
jsonl_path = "/mnt/pvc/results/monitor_metrics.jsonl"

pd.DataFrame(metrics_buffer).to_csv(csv_path, index=False)
with open(jsonl_path, "w") as fp:
    for e in metrics_buffer:
        fp.write(json.dumps(e) + "\n")

with open(csv_path, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.csv", fp.read(), "text/csv")
with open(jsonl_path, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.jsonl", fp.read(), "application/json")

sync_all_metrics_to_minio()

# —— 写入 KFP metadata ——  
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[monitor] metrics uploaded – bye.")
