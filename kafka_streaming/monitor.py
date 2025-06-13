#!/usr/bin/env python3
"""
kafka_streaming/monitor.py ─ 真·流式 Drift 监控 (Window-based)
──────────────────────────────────────────────────────────────
• 每收到 1 条 Kafka 消息立即放入滑动窗口  
• 窗口长度达到 WINDOW_SIZE 时计算一次 Jensen–Shannon  
• 超过 JS_TRIGGER_THRESH 即刻触发 dynamic_retrain  
• retrain 完成后把当前窗口并入 baseline，JS 立即回落  
• log_metric() → 本地 CSV/JSONL → 统一上传 MinIO
"""

from __future__ import annotations
import os, json, queue, threading, time, sys, subprocess          # ★ subprocess 改非阻塞
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
WINDOW_SIZE          = int(os.getenv("JS_WINDOW_SIZE", "50"))
MIN_RETRAIN_INTERVAL = int(os.getenv("MIN_RETRAIN_INTERVAL", "300"))  # ★ 5 min 冷却
metrics_buffer : list[dict] = []
producer_done  = threading.Event()
q: queue.Queue = queue.Queue()
retrain_lock   = threading.Lock()                                     # ★ 同时只跑 1 个
retrain_running = False                                              # ★ 状态位

# —— 预加载 baseline（离线训练见过的分布）——
BASELINE_KEY = os.getenv("BASELINE_KEY", "datasets_old/old_total.csv")
baseline_df  = load_csv(BASELINE_KEY)
baseline_df.drop(columns=["input_rate", "latency", "output_rate"],
                 errors="ignore", inplace=True)
baseline_df  = baseline_df.reindex(columns=FEATURE_COLS, fill_value=0.0)
JS_FEATS     = [f for f in FEATURE_COLS if f in baseline_df.columns]

# ----------------------------------------------------------------------
# 1. 轻量埋点
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
    open("/mnt/pvc/results/monitor_ready.flag","w").close()
    print("[monitor] readiness flag touched")

    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            producer_done.set(); continue
        q.put(v)

threading.Thread(target=_listener, daemon=True).start()

# ----------------------------------------------------------------------
# 3. JS 计算
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
# 4. retrain 触发辅助
# ----------------------------------------------------------------------
def _bg_retrain(js_val: float, snapshot: pd.DataFrame):
    """后台串行 retrain + baseline 扩充"""
    global retrain_running, baseline_df
    with retrain_lock: retrain_running = True
    cmd = ["python","-m","ml.dynamic_retrain",f"{js_val:.4f}"]
    print(f"[monitor] ➜ exec: {' '.join(cmd)}")
    subprocess.call(cmd)                        # ★ 同步等待
    baseline_df = pd.concat([baseline_df, snapshot], ignore_index=True)  # ★ baseline 扩充
    with retrain_lock: retrain_running = False
    print("[monitor] retrain done & baseline updated")

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
            print("[monitor] producer_done & queue empty → bye"); break
        continue

    msg_count += 1
    win_rows.append(item)
    if len(win_rows) > WINDOW_SIZE: win_rows.pop(0)
    if len(win_rows) < WINDOW_SIZE: continue

    t0 = time.time()
    df_win = pd.DataFrame([r["features"] for r in win_rows])
    js_val = _avg_js(df_win)
    js_ms  = (time.time()-t0)*1000

    record_metric("monitor","drift_calc",
                  js_val=round(js_val,4), window_size=WINDOW_SIZE,
                  msg_since_start=msg_count, runtime_ms=round(js_ms,3),
                  **_sys_metrics())

    print(f"[monitor] JS={js_val:.4f} (thr={JS_TRIGGER_THRESH}) msgs={msg_count}")

    if (js_val > JS_TRIGGER_THRESH
        and (time.time()-last_retrain_ts) >= MIN_RETRAIN_INTERVAL
        and not retrain_running):
        last_retrain_ts = time.time()
        snapshot = df_win.copy()

        # ★ 判断 severity
        if js_val > JS_SEV2_THRESH:
            severity = "K"
        elif js_val > JS_SEV1_THRESH:
            severity = "2"
        else:
            severity = "1"

        print(f"[monitor] retrain triggered by JS={js_val:.4f} → Severity-{severity}")

        threading.Thread(target=_bg_retrain,
                         args=(js_val, snapshot),
                         daemon=True).start()

        record_metric("monitor", "retrain_trigger", js_val=round(js_val, 4), severity=severity)


# ----------------------------------------------------------------------
# 6. 收尾：写指标 & 上传
# ----------------------------------------------------------------------
print("[monitor] writing buffered metrics …")
os.makedirs("/mnt/pvc/results", exist_ok=True)
csv_path   = "/mnt/pvc/results/monitor_metrics.csv"
jsonl_path = "/mnt/pvc/results/monitor_metrics.jsonl"

pd.DataFrame(metrics_buffer).to_csv(csv_path,index=False)
with open(jsonl_path,"w") as fp:
    for e in metrics_buffer: fp.write(json.dumps(e)+"\n")

with open(csv_path,"rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.csv", fp.read(),"text/csv")
with open(jsonl_path,"rb") as fp:
    save_bytes(f"{RESULT_DIR}/monitor_metrics.jsonl", fp.read(),"application/json")

sync_all_metrics_to_minio()

open("/tmp/kfp_outputs/output_metadata.json","w").write("{}")
print("[monitor] metrics uploaded – bye.")
