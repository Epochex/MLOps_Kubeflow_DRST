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
from kafka import KafkaConsumer, KafkaProducer
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

# 0. 全局 & 初始化
# 动态的后10%combined 数据+300条训练的数据(一开始的时候就是约500多条combined数据)作为基线数据，然后跟滑动窗口为200的流数据进行js漂移值计算，
# 一旦触发retrain，就把基线数据中300条训练的数据立刻替换retrain前的300条流数据，但仍然保留10%combined 数据，也就是10%combined 数据+300条retrain前最新数据
# —— 窗口大小 —— 
WINDOW_SIZE          = 300
# 初始到达多少条立刻强制 retrain
NEW_SAMPLE_MIN       = int(os.getenv("NEW_SAMPLE_MIN", "100"))
MIN_RETRAIN_INTERVAL = float(os.getenv("MIN_RETRAIN_INTERVAL", "0.1"))

infer_msg_count = 0    # 累计三个 inference consumer 处理的消息数
first_forced = False   # 记录是否已经做过第一次强制 retrain

# —— 声明指标 buffer、Producer 完成 flag、消息队列 q ——  
metrics_buffer: list[dict] = []
producer_done  = threading.Event()
q: queue.Queue = queue.Queue()

# >>>  Drift re-training state  <<<
retrain_buf:   list[dict]     = []          # 最近收到的所有样本
retrain_lock:  threading.Lock = threading.Lock()
retrain_running: bool         = False       # 后台重训是否正在进行

# —— 基线动态配置 —— 
TRAIN_N             = 500
BASELINE_KEY        = os.getenv("BASELINE_KEY", "datasets/combined.csv")

# 1) 从 MinIO 取 combined.csv 并只保留 FEATURE_COLS
combined_df = load_csv(BASELINE_KEY)
combined_df.drop(columns=["input_rate", "latency", "output_rate"],
                 errors="ignore", inplace=True)
combined_df = combined_df.reindex(columns=FEATURE_COLS, fill_value=0.0)

# 2) 拿尾部 10% 作为 static 基线
ten_n = TRAIN_N
base_combined = combined_df.tail(TRAIN_N).reset_index(drop=True)

# ✅ 3) 初始化基线：只保留 static 部分（尾部10%）
baseline_df = base_combined.copy()

# 全部 60 维都参与 JS 比较
JS_FEATS = FEATURE_COLS.copy()


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
            check=True
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

    except Exception as exc:
        print(f"[monitor] retrain thread error: {exc}")

    finally:
        # 4) 无论成功失败，都在这里更新触发时间 & 重置状态
        last_retrain_ts = time.time()
        retrain_running = False

# ----------------------------------------------------------------------
# 5. 主循环（更新后）
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
        continue  # 继续等待

    msg_count += 1
    win_rows.append(item)
    retrain_buf.append(item)

    if not first_forced and infer_msg_count >= NEW_SAMPLE_MIN:
        first_forced = True
        print(f"[monitor] initial force retrain at infer_msg_count={infer_msg_count}")
        # 1) 广播暂停
        ctrl_prod.send(CONTROL_TOPIC, {"retrain": "start"})
        # 2) 阻断式 retrain
        _bg_retrain(JS_TRIGGER_THRESH, retrain_buf.copy())
        # 3) 广播恢复
        ctrl_prod.send(CONTROL_TOPIC, {"retrain": "end"})

    # 保持窗口固定长度 & 未满则跳过
    if len(win_rows) > WINDOW_SIZE:
        win_rows.pop(0)
    if len(win_rows) < WINDOW_SIZE:
        continue

    # 计算 JS 值
    t0 = time.time()
    df_win = pd.DataFrame([r["features"] for r in win_rows])
    js_val = _avg_js(df_win)
    js_ms  = (time.time() - t0) * 1000

    record_metric("monitor", "drift_calc",
                  js_val=round(js_val, 4), window_size=WINDOW_SIZE,
                  msg_since_start=msg_count, runtime_ms=round(js_ms, 3),
                  **_sys_metrics())

    print(f"[monitor] JS={js_val:.4f} (thr={JS_TRIGGER_THRESH}) msgs={msg_count}")

    # ───── 触发重训 ─────────────────────────────────────────────
    now = time.time()
    if (js_val > JS_TRIGGER_THRESH
        and (now - last_retrain_ts) >= MIN_RETRAIN_INTERVAL
        and not retrain_running):

        # —— 阻断式 retrain ——
        retrain_running = True
        snapshot = retrain_buf[-max(TRAIN_N, WINDOW_SIZE):]

        # 1) 通知下游：暂停生产/消费
        ctrl_prod.send(CONTROL_TOPIC, {"retrain": "start"})

        # 2) 同步调用 retrain（内部会更新 baseline_df、last_retrain_ts 并重置 retrain_running）
        _bg_retrain(js_val, snapshot)

        # 3) retrain 完毕，通知下游恢复
        ctrl_prod.send(CONTROL_TOPIC, {"retrain": "end"})

        # 4) 记录触发事件
        record_metric("monitor", "retrain_trigger",
                    js_val=round(js_val, 4),
                    severity=(
                        "K" if js_val > JS_SEV2_THRESH else
                        "2" if js_val > JS_SEV1_THRESH else
                        "1"
                    ))


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
