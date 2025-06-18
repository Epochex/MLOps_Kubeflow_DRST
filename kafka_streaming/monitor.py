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

from collections import deque



# 0. 全局 & 初始化
# 动态的后10% combined 数据 + 300 条训练数据（初始约 500 多条 combined 数据）作为基线，
# 用滑动窗口为 200 的流数据计算 JS 漂移，一旦触发 retrain 就用最新 300 条流数据替换原有训练数据。
# —— 窗口大小 —— 
WINDOW_SIZE          = 300
# 初始达到多少条流数据后立即强制 retrain
NEW_SAMPLE_MIN       = int(os.getenv("NEW_SAMPLE_MIN", "100"))
# 两次 retrain 最小时间间隔（秒）
MIN_RETRAIN_INTERVAL = float(os.getenv("MIN_RETRAIN_INTERVAL", "0.1"))

# —— 静态训练数据大小 —— 
TRAIN_N              = 500

# —— 缓冲区长度：取 TRAIN_N 和 WINDOW_SIZE 的最大值，自动丢弃最老样本 —— 
from collections import deque
MAX_BUF = max(TRAIN_N, WINDOW_SIZE)
retrain_buf: deque[dict] = deque(maxlen=MAX_BUF)

# —— 重训练状态标志 —— 
import threading, queue
retrain_lock    = threading.Lock()
retrain_running = False       # 后台重训是否正在进行

# —— 统计 inference consumer 处理总量，用于初始强制 retrain —— 
infer_msg_count = 0
first_forced    = False       # 是否已做过第一次强制 retrain

# —— 指标 buffer、producer_done flag、消息队列 q —— 
metrics_buffer: list[dict] = []
producer_done            = threading.Event()
q: queue.Queue           = queue.Queue()

# —— 基线数据加载 —— 
import os
BASELINE_KEY = os.getenv("BASELINE_KEY", "datasets/combined.csv")
combined_df  = load_csv(BASELINE_KEY)
combined_df.drop(columns=["input_rate", "latency", "output_rate"],
                 errors="ignore", inplace=True)
combined_df = combined_df.reindex(columns=FEATURE_COLS, fill_value=0.0)

# 取尾部 TRAIN_N 条作为 static 基线
ten_n         = TRAIN_N
base_combined = combined_df.tail(TRAIN_N).reset_index(drop=True)
baseline_df   = base_combined.copy()

# 所有 60 维特征用于 JS 计算
JS_FEATS = FEATURE_COLS.copy()

# # —— 基线动态配置 动态混合—— 
# STATIC_N             = 150
# DYNAMIC_N            = 150
# BASELINE_KEY         = os.getenv("BASELINE_KEY", "datasets/combined.csv")

# # 1) 从 MinIO 取 combined.csv 并只保留 FEATURE_COLS
# combined_df = load_csv(BASELINE_KEY)
# combined_df.drop(columns=["input_rate", "latency", "output_rate"],
#                  errors="ignore", inplace=True)
# combined_df = combined_df.reindex(columns=FEATURE_COLS, fill_value=0.0)

# # 2) 拿尾部 STATIC_N 条作为 static 基线
# base_combined = combined_df.tail(STATIC_N).reset_index(drop=True)

# # ✅ 3) 初始化基线：只保留 static 部分
# baseline_df = base_combined.copy()

# # 全部 60 维都参与 JS 比较
# JS_FEATS = FEATURE_COLS.copy()



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
TMP_DIR = "/tmp/monitor"                # ← 改用 /tmp 做临时存储
os.makedirs(TMP_DIR, exist_ok=True)


def _listener():
    global infer_msg_count
    
    
    cons = KafkaConsumer(
        KAFKA_TOPIC,                 # 原来的拉流 topic
        KAFKA_TOPIC + "_infer_count",# 新增：各 consumer 发过来的 processed 计数
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-monitor",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )

    # —— readiness flag：本地 /tmp + MinIO ————————————————
    flag_local = os.path.join(TMP_DIR, "monitor_ready.flag")
    open(flag_local, "w").close()
    print("[monitor] readiness flag touched →", flag_local)
    save_bytes(f"{RESULT_DIR}/monitor_ready.flag", b"", "text/plain")

    # 持续 poll()，并捕获所有异常后重试
    while True:
        try:
            records = cons.poll(timeout_ms=1000)
            for tp, msgs in records.items():
                for msg in msgs:
                    v = msg.value
                    if v.get("producer_done"):
                        producer_done.set()
                    # 新增： inference consumer 发过来的 processed 计数
                    elif "processed" in v:
                        infer_msg_count += int(v["processed"])
                    else:
                        q.put(v)
        except Exception as e:
            # 打印日志但保持线程存活
            print(f"[monitor] listener error, restarting poll(): {e}")
            time.sleep(1)
            continue

# 启动监听线程（daemon 模式，不阻塞主线程）
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
# 4. 重训后台线程（更新后）
# ----------------------------------------------------------------------
def _bg_retrain(js_val: float, snapshot_rows: list[dict]):
    """
    ① 把当前窗口写入 /tmp/monitor/latest_batch.npy（供 dynamic_retrain）
    ② 同步调用 dynamic_retrain，捕获成功或失败
    ③ 成功时扩充 baseline_df；失败时跳过更新
    ④ 在 finally 中更新 last_retrain_ts 并重置 retrain_running
    """
    global retrain_running, baseline_df, last_retrain_ts

    try:
        print(f"[monitor] retrain start (JS={js_val:.4f})")

        # 1) Dump 最新窗口到本地临时目录
        local_np = os.path.join(TMP_DIR, "latest_batch.npy")
        np.save(local_np, np.array(snapshot_rows, dtype=object))
        print(f"[monitor] latest_batch.npy saved → {local_np}")

        # 2) 真正运行动态重训
        subprocess.run(
            ["python", "-m", "ml.dynamic_retrain", str(js_val)],
            check=True,
            timeout=600
        )
        print("[monitor] dynamic_retrain finished")

        # 3) 只有 retrain 成功后，才更新 baseline_df
        with retrain_lock:
            combined_part = baseline_df.iloc[:ten_n].copy()
            recent_features = [r["features"] for r in snapshot_rows[-TRAIN_N:]]
            train_part = (
                pd.DataFrame(recent_features)
                  .reindex(columns=FEATURE_COLS, fill_value=0.0)
            )
            baseline_df = pd.concat([combined_part, train_part], ignore_index=True)
        #3) 只有 retrain 成功后，才更新 baseline_df：static + dynamic 合并
        
        # with retrain_lock:
        #     # static：始终使用原始 combined 最后 STATIC_N 条
        #     combined_part = base_combined.copy()
        #     # dynamic：使用最新 DYNAMIC_N 条流数据
        #     recent_features = [r["features"] for r in snapshot_rows[-DYNAMIC_N:]]
        #     train_part = (
        #         pd.DataFrame(recent_features)
        #         .reindex(columns=FEATURE_COLS, fill_value=0.0)
        #     )
        #     baseline_df = pd.concat([combined_part, train_part], ignore_index=True)


    except Exception as exc:
        print(f"[monitor] retrain thread error: {exc}")

    finally:
        # 4) 无论成功失败，都在这里更新触发时间 & 重置状态
        last_retrain_ts = time.time()
        retrain_running = False

# ----------------------------------------------------------------------
# 5. 主循环（更新后：初始快速重训使用最新 NEW_SAMPLE_MIN 条流数据）
# ----------------------------------------------------------------------
print(f"[monitor] start, WINDOW={WINDOW_SIZE}, THR={JS_TRIGGER_THRESH}")
win_rows: list[dict] = []
retrain_buf: deque = deque(maxlen=MAX_BUF)
last_retrain_ts = 0.0
msg_count = 0
infer_msg_count = 0
first_forced = False

while True:
    try:
        item = q.get(timeout=CONSUME_IDLE_S)
    except queue.Empty:
        if producer_done.is_set():
            print("[monitor] producer_done & queue empty → bye")
            break
        continue  # 继续等待

    msg_count += 1
    infer_msg_count += 1
    win_rows.append(item)
    retrain_buf.append(item)

    # —— 初始快速重训：仅用最新 NEW_SAMPLE_MIN 条流数据 ——
    if not first_forced and infer_msg_count >= NEW_SAMPLE_MIN:
        first_forced = True
        print(f"[monitor] initial quick-retrain at infer_msg_count={infer_msg_count}")
        retrain_running = True

        # 最新的 NEW_SAMPLE_MIN 条样本（deque 不支持 slice，需要先转 list）
        snapshot = list(retrain_buf)[-NEW_SAMPLE_MIN:]

        threading.Thread(
            target=_bg_retrain,
            args=(JS_TRIGGER_THRESH, snapshot),
            daemon=True
        ).start()

    # —— 滑动窗口固定长度 —— 
    if len(win_rows) > WINDOW_SIZE:
        win_rows.pop(0)
    if len(win_rows) < WINDOW_SIZE:
        continue

    # —— 计算 JS 距离 ——
    t0 = time.time()
    df_win = pd.DataFrame([r["features"] for r in win_rows])
    js_val = _avg_js(df_win)
    js_ms  = (time.time() - t0) * 1000

    record_metric("monitor", "drift_calc",
                  js_val=round(js_val, 4), window_size=WINDOW_SIZE,
                  msg_since_start=msg_count, runtime_ms=round(js_ms, 3),
                  **_sys_metrics())

    print(f"[monitor] JS={js_val:.4f} (thr={JS_TRIGGER_THRESH}) msgs={msg_count}")

    # —— 达到阈值 & 不在冷却期 & 没有 retrain 正在运行 —— 
    now = time.time()
    if (js_val > JS_TRIGGER_THRESH
        and (now - last_retrain_ts) >= MIN_RETRAIN_INTERVAL
        and not retrain_running):

        retrain_running = True
        snapshot = list(retrain_buf)[-max(TRAIN_N, WINDOW_SIZE):]

        # 判断严重度
        if   js_val > JS_SEV2_THRESH: severity = "K"
        elif js_val > JS_SEV1_THRESH: severity = "2"
        else:                          severity = "1"

        threading.Thread(
            target=_bg_retrain,
            args=(js_val, snapshot),
            daemon=True
        ).start()

        record_metric("monitor", "retrain_trigger",
                      js_val=round(js_val, 4), severity=severity)


# ----------------------------------------------------------------------
# 6. 收尾：写指标 & 上传
# ----------------------------------------------------------------------
print("[monitor] writing buffered metrics …")

local_res = os.path.join(TMP_DIR, "results")
os.makedirs(local_res, exist_ok=True)

csv_path   = os.path.join(local_res, "monitor_metrics.csv")
jsonl_path = os.path.join(local_res, "monitor_metrics.jsonl")

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
