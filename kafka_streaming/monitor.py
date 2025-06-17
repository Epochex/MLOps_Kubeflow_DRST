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

# ────────────────────────── 常量 & 全局状态 ───────────────────────────
WINDOW_SIZE          = 200   # 滑动窗口大小
MIN_RETRAIN_INTERVAL = float(os.getenv("MIN_RETRAIN_INTERVAL", "0.1"))
TRAIN_N              = int(os.getenv("TRAIN_N", "500"))  # 实际传给重训练的总样本数，但训练用多少条看dynamic代码里面
BASELINE_KEY         = os.getenv("BASELINE_KEY", "datasets/combined.csv")

# 指标 buffer、生产者完成标志、本地消息队列
metrics_buffer: list[dict] = []
producer_done = threading.Event()
q: queue.Queue = queue.Queue()

# retrain 状态
retrain_buf: list[dict]     = []
retrain_lock               = threading.Lock()
retrain_running: bool      = False

# ────────────────────────── 加载基线数据 ────────────────────────────
combined_df = load_csv(BASELINE_KEY)
combined_df = combined_df.drop(columns=["input_rate", "latency", "output_rate"],
                               errors="ignore")
combined_df = combined_df.reindex(columns=FEATURE_COLS, fill_value=0.0)
baseline_df = combined_df.tail(min(TRAIN_N, len(combined_df))).reset_index(drop=True)
JS_FEATS    = FEATURE_COLS.copy()

# ────────────────────────── 指标工具 ───────────────────────────────
_proc = psutil.Process()
def record_metric(component: str, event: str, **kw):
    entry = {
        "utc": datetime.utcnow().isoformat() + "Z",
        "component": component,
        "event": event,
        **kw
    }
    metrics_buffer.append(entry)
    log_metric(component=component, event=event, **kw)

def _sys_metrics() -> dict:
    t = psutil.cpu_times_percent(interval=None)
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "mem_pct": psutil.virtual_memory().percent,
        "io_wait_pct": getattr(t, "iowait", 0.0),
    }

# ────────────────────────── Kafka 监听线程 ───────────────────────────
TMP_DIR = "/tmp/monitor"
os.makedirs(TMP_DIR, exist_ok=True)

def _listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-monitor",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    # readiness flag
    local_flag = os.path.join(TMP_DIR, "monitor_ready.flag")
    open(local_flag, "w").close()
    print(f"[monitor] readiness flag → {local_flag}")
    save_bytes(f"{RESULT_DIR}/monitor_ready.flag", b"", "text/plain")

    while True:
        try:
            records = cons.poll(timeout_ms=1000)
            for tp, msgs in records.items():
                for msg in msgs:
                    v = msg.value
                    if v.get("producer_done"):
                        producer_done.set()
                    else:
                        q.put(v)
        except Exception as e:
            print(f"[monitor] listener error, restarting poll(): {e}")
            time.sleep(1)

threading.Thread(target=_listener, daemon=True).start()

# ────────────────────────── Jensen–Shannon ─────────────────────────
def _js(p: np.ndarray, q_: np.ndarray, bins=50, eps=1e-9) -> float:
    p = p[~np.isnan(p)]; q_ = q_[~np.isnan(q_)]
    if p.size == 0 or q_.size == 0:
        return 0.0
    comb = np.concatenate([p, q_])
    lo, hi = np.percentile(comb, [0.5, 99.5])
    if lo >= hi:
        return 0.0
    hp, _ = np.histogram(p,  bins=bins, range=(lo,hi))
    hq, _ = np.histogram(q_, bins=bins, range=(lo,hi))
    hp = hp.astype(float) + eps; hq = hq.astype(float) + eps
    hp /= hp.sum();      hq /= hq.sum()
    return float(jensenshannon(hp, hq))

def _avg_js(df: pd.DataFrame) -> float:
    return float(np.mean([
        _js(baseline_df[f].values, df[f].values) for f in JS_FEATS
    ])) if JS_FEATS else 0.0

# ────────────────────────── 重训后台线程 ───────────────────────────
def _bg_retrain(js_val: float, snapshot: list[dict]):
    global retrain_running, baseline_df
    try:
        print(f"[monitor] retrain start (JS={js_val:.4f}), rows={len(snapshot)}")
        with retrain_lock:
            retrain_running = True

        path_np = os.path.join(TMP_DIR, "latest_batch.npy")
        np.save(path_np, np.array(snapshot, dtype=object))
        print(f"[monitor] saved latest_batch.npy → {path_np}")

        cmd = [sys.executable, "-m", "ml.dynamic_retrain", f"{js_val:.4f}"]
        print(f"[monitor] executing retrain: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate()
        print(f"[monitor] retrain stdout:\n{out}")
        if err:
            print(f"[monitor] retrain stderr:\n{err}")
        if proc.returncode != 0:
            print(f"[monitor] retrain FAILED (exit {proc.returncode})")
            return

        print("[monitor] retrain succeeded, updating baseline_df")
        with retrain_lock:
            baseline_df = (
                pd.DataFrame(snapshot[-WINDOW_SIZE:])
                  .reindex(columns=FEATURE_COLS, fill_value=0.0)
            )

    except Exception as e:
        print(f"[monitor] retrain exception: {e}")
    finally:
        with retrain_lock:
            retrain_running = False
        print("[monitor] retrain thread exit")

# ────────────────────────── 主循环 ────────────────────────────────
print(f"[monitor] start: WINDOW={WINDOW_SIZE}, THR={JS_TRIGGER_THRESH}")
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
    win_rows.append(item)
    retrain_buf.append(item)

    if len(win_rows) > WINDOW_SIZE:
        win_rows.pop(0)
    if len(win_rows) < WINDOW_SIZE:
        continue

    t0 = time.time()
    df_win = pd.DataFrame([r["features"] for r in win_rows])
    js_val = _avg_js(df_win)
    js_ms  = (time.time() - t0) * 1000

    record_metric(
        "monitor", "drift_calc",
        js_val=round(js_val,4),
        window_size=WINDOW_SIZE,
        msg_since_start=msg_count,
        runtime_ms=round(js_ms,3),
        **_sys_metrics()
    )
    print(f"[monitor] JS={js_val:.4f} (thr={JS_TRIGGER_THRESH}) msgs={msg_count}")

    now = time.time()
    if (js_val > JS_TRIGGER_THRESH
        and (now - last_retrain_ts) >= MIN_RETRAIN_INTERVAL
        and not retrain_running):

        last_retrain_ts = now
        snapshot = retrain_buf[-max(TRAIN_N, WINDOW_SIZE):]

        if   js_val > JS_SEV2_THRESH: sev = "K"
        elif js_val > JS_SEV1_THRESH: sev = "2"
        else:                          sev = "1"

        threading.Thread(
            target=_bg_retrain,
            args=(js_val, snapshot),
            daemon=True
        ).start()

        record_metric("monitor", "retrain_trigger",
                      js_val=round(js_val,4), severity=sev)

# ────────────────────────── 收尾：写指标 & 上传 ───────────────────────
print("[monitor] writing buffered metrics …")
local_res = os.path.join(TMP_DIR, "results")
os.makedirs(local_res, exist_ok=True)

csv_path   = os.path.join(local_res, "monitor_metrics.csv")
jsonl_path = os.path.join(local_res, "monitor_metrics.jsonl")

pd.DataFrame(metrics_buffer).to_csv(csv_path, index=False)
with open(jsonl_path, "w") as fp:
    for e in metrics_buffer:
        fp.write(json.dumps(e) + "\n")

save_bytes(f"{RESULT_DIR}/monitor_metrics.csv", open(csv_path,"rb").read(), "text/csv")
save_bytes(f"{RESULT_DIR}/monitor_metrics.jsonl", open(jsonl_path,"rb").read(), "application/json")
sync_all_metrics_to_minio()

# 写入 KFP metadata
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[monitor] metrics uploaded – bye.")