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

# 0. 全局 & 初始化
# 动态的后10%combined 数据+300条训练的数据(一开始的时候就是约500多条combined数据)作为基线数据，然后跟滑动窗口为200的流数据进行js漂移值计算，
# 一旦触发retrain，就把基线数据中300条训练的数据立刻替换retrain前的300条流数据，但仍然保留10%combined 数据，也就是10%combined 数据+300条retrain前最新数据
# —— 窗口大小 —— 
WINDOW_SIZE          = 200
RETRAIN_BATCH_SIZE   = int(os.getenv("RETRAIN_BATCH_SIZE", "500"))
MIN_RETRAIN_INTERVAL = int(os.getenv("MIN_RETRAIN_INTERVAL", "3"))

# —— 声明指标 buffer、Producer 完成 flag、消息队列 q ——  
metrics_buffer: list[dict] = []
producer_done  = threading.Event()
q: queue.Queue = queue.Queue()

# >>>  Drift re-training state  <<<
retrain_buf:   list[dict]     = []          # 最近收到的所有样本
retrain_lock:  threading.Lock = threading.Lock()
retrain_running: bool         = False       # 后台重训是否正在进行

# —— 基线动态配置 —— 
#  保留 combined.csv 的后10%  + 最新500条流数据
PCT                 = 0.1
TRAIN_N             = 500
BASELINE_KEY        = os.getenv("BASELINE_KEY", "datasets/combined.csv")

# 1) 从 MinIO 取 combined.csv 并只保留 FEATURE_COLS
combined_df = load_csv(BASELINE_KEY)
combined_df.drop(columns=["input_rate", "latency", "output_rate"],
                 errors="ignore", inplace=True)
combined_df = combined_df.reindex(columns=FEATURE_COLS, fill_value=0.0)

# 2) 拿尾部 10%
ten_n = max(1, int(PCT * len(combined_df)))
base_combined = combined_df.tail(ten_n).reset_index(drop=True)

# 3) 初始训练池：从 combined_df 抽样 300 条（若未来要用离线 artefacts，可改为加载文件）
init_train = combined_df.sample(n=min(TRAIN_N, len(combined_df)),
                                random_state=0).reset_index(drop=True)

# 4) 合并成 baseline_df，用于 JS 计算
baseline_df = pd.concat([base_combined, init_train], ignore_index=True)

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
    cons = KafkaConsumer(
        KAFKA_TOPIC,
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
# 4. 重训后台线程
# ----------------------------------------------------------------------
def _bg_retrain(js_val: float, snapshot_rows: list[dict]):
    """
    ① 把当前窗口写入 /tmp/monitor/latest_batch.npy（供 dynamic_retrain）
    ② 同步调用 dynamic_retrain，捕获成功或失败
    ③ 成功时扩充 baseline_df；失败时重置状态，跳过更新
    """
    global retrain_running, baseline_df
    try:
        print(f"[monitor] retrain start (JS={js_val:.4f})")

        # 1) Dump 最新窗口到本地临时目录
        local_np = os.path.join(TMP_DIR, "latest_batch.npy")
        np.save(local_np, np.array(snapshot_rows, dtype=object))
        print(f"[monitor] latest_batch.npy saved → {local_np}")

        # 2) 同步调用 retrain 子进程，并捕获异常
        with retrain_lock:
            retrain_running = True
        cmd = ["python", "-m", "ml.dynamic_retrain", f"{js_val:.4f}"]
        print("[monitor] ➜ executing retrain:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print("[monitor] retrain succeeded")
        except subprocess.CalledProcessError as e:
            print(f"[monitor] retrain failed: exit {e.returncode}")
            with retrain_lock:
                retrain_running = False
            return

        # 3) 只有 retrain 成功后，才更新 baseline_df
        with retrain_lock:
            combined_part = baseline_df.iloc[:ten_n].copy()
            recent_features = [r["features"] for r in snapshot_rows[-TRAIN_N:]]
            train_part = (
                pd.DataFrame(recent_features)
                  .reindex(columns=FEATURE_COLS, fill_value=0.0)
            )
            baseline_df = pd.concat([combined_part, train_part], ignore_index=True)
            retrain_running = False

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
        # 无消息且生产者未完成，继续等待
        continue

    msg_count += 1
    win_rows.append(item)            # 加入滑动窗口
    retrain_buf.append(item)         # 缓存用于 dynamic_retrain

    # 保持窗口固定长度
    if len(win_rows) > WINDOW_SIZE:
        win_rows.pop(0)
    if len(win_rows) < WINDOW_SIZE:
        continue  # 窗口未满，跳过 JS 计算

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

        last_retrain_ts = now

        # ✅ 只用“当前滑动窗口”做 snapshot
        snapshot = retrain_buf[-max(TRAIN_N, WINDOW_SIZE):]

        # 判断严重度 …
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
