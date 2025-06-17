#!/usr/bin/env python3
"""
kafka_streaming/producer.py — Kafka 数据生产者，支持接收 retrain 控制消息暂停／恢复
"""
import os
import sys
import time
import json
import glob
import datetime
import threading

import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer

# 把项目根目录放进 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.minio_helper import s3, BUCKET, load_csv, save_bytes
from shared.config       import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    TARGET_COL, BATCH_SIZE, RESULT_DIR, CONTROL_TOPIC
)
from shared.features      import FEATURE_COLS
from shared.metric_logger import log_metric, sync_all_metrics_to_minio

# ---------- 发送节奏 ----------
SLEEP = float(os.getenv("MSG_SLEEP", "0.5"))
LIM1  = int(os.getenv("LIMIT_PHASE1", "500"))
LIM2  = int(os.getenv("LIMIT_PHASE2", "500"))
LIM3  = int(os.getenv("LIMIT_PHASE3", "250"))

STAGES = [
    ("Phase-1", f"datasets/random_rates.csv", LIM1),
    ("Phase-2", f"datasets/resource_stimulus_global_A-B-C_modified.csv", LIM2),
    ("Phase-3", f"datasets/resource_stimulus_global_A-B-C_modified.csv", LIM3),
]

# ---------- 本地缓存 & 时间戳 ----------
TMP_DIR = "/tmp/producer"
os.makedirs(TMP_DIR, exist_ok=True)
_START = datetime.datetime.utcnow()
FLAG_DONE_LOCAL = os.path.join(TMP_DIR, "producer_done.flag")

# ---------- 控制暂停/恢复 的全局 event ----------
pause_event = threading.Event()

def _ctrl_listener():
    """监听 CONTROL_TOPIC 上的 retrain:start/end 消息"""
    cons = KafkaConsumer(
        CONTROL_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    for msg in cons:
        v = msg.value
        if v.get("retrain") == "start":
            print("[producer] ⏸ pausing send (retrain start)")
            pause_event.set()
        elif v.get("retrain") == "end":
            print("[producer] ▶ resuming send (retrain end)")
            pause_event.clear()

# 启动控制监听线程
threading.Thread(target=_ctrl_listener, daemon=True).start()

# ---------- Helper ----------
def _ensure_local(rel_path: str) -> str:
    abs_path = os.path.abspath(rel_path)
    if os.path.exists(abs_path):
        return abs_path
    print(f"[producer] ▼ download {rel_path} from MinIO …")
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    obj = s3.get_object(Bucket=BUCKET, Key=rel_path)
    with open(abs_path, "wb") as fp:
        fp.write(obj["Body"].read())
    print(f"[producer]   saved → {abs_path}")
    return abs_path

def send_df(df: pd.DataFrame, prod: KafkaProducer, phase: str):
    """
    加随机 key = os.urandom(4)，确保 round-robin 分区。
    支持在 retrain 期间自动 pause。
    """
    total, sent, batch_id = len(df), 0, 0
    for idx, row in df.iterrows():
        # —— 如果正在重训，则暂停发送 —— 
        while pause_event.is_set():
            time.sleep(0.5)

        payload = {
            "phase": phase,
            "row_index": int(idx),
            "features": {c: float(row[c]) for c in FEATURE_COLS},
            "label": float(row[TARGET_COL]),
            "send_ts": datetime.datetime.utcnow().isoformat() + "Z"
        }
        prod.send(
            KAFKA_TOPIC,
            key=os.urandom(4),
            value=payload
        )

        sent += 1
        if sent % BATCH_SIZE == 0 or sent == total:
            prod.flush()
            batch_id += 1
            print(f"[producer] {phase} | batch {batch_id:03d} | {sent}/{total}")
        time.sleep(SLEEP)

    log_metric(component="producer", event="stage_done",
               phase=phase, rows=int(total))
    print(f"[producer] ✓ finished {phase}")

# ---------- main ----------
def main():
    print("[producer] start …")
    
    time.sleep(10)
    save_bytes(f"{RESULT_DIR}/producer_ready.flag", b"", "text/plain")
    # —— 等 monitor 写入 MinIO readiness flag ——  
    key = f"{RESULT_DIR}/monitor_ready.flag"
    print(f"[producer] polling MinIO for {key} …")
    while True:
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            break
        except:
            time.sleep(1)
    print("[producer] monitor ready → start producing")

    # —— 等 consumer 准备好 ——  
    num_consumers = int(os.getenv("NUM_CONSUMERS", "3"))
    timeout       = int(os.getenv("CONSUMER_WAIT_TIMEOUT", "60"))
    prefix        = f"{RESULT_DIR}/consumer_ready_"
    print(f"[producer] polling MinIO for {prefix}*.flag …")
    t0 = time.time()
    ready_count = 0
    while time.time() - t0 < timeout:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        contents = resp.get("Contents", [])
        ready_count = len([o for o in contents if o["Key"].endswith(".flag")])
        if ready_count >= num_consumers:
            break
        time.sleep(1)
    print(f"[producer] detected {ready_count}/{num_consumers} consumers ready → continue")

    # —— KafkaProducer ——  
    prod = KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=lambda m: json.dumps(m).encode(),
    )

    # —— 逐阶段发送 ——  
    for phase, rel_path, limit in STAGES:
        df_path = _ensure_local(rel_path)
        df = (pd.read_csv(df_path, index_col=0)
                .replace({'<not counted>': np.nan})
                .dropna()
                .reset_index(drop=True))
        if limit:
            df = df.iloc[:limit]
        send_df(df, prod, phase)

    # —— sentinel ——  
    sentinel = {
        "producer_done": True,
        "send_ts": datetime.datetime.utcnow().isoformat() + "Z"
    }
    partitions = prod.partitions_for(KAFKA_TOPIC)
    for p in partitions:
        prod.send(KAFKA_TOPIC, partition=p, value=sentinel)
    prod.flush()
    prod.close()

    # 本地 & MinIO 标记
    open(FLAG_DONE_LOCAL, "w").close()
    save_bytes(f"{RESULT_DIR}/producer_done.flag", b"", "text/plain")
    print(f"[producer] sent sentinel to {len(partitions)} partitions")

    # timing & metrics
    timing_dir = os.path.join(TMP_DIR, "timing")
    os.makedirs(timing_dir, exist_ok=True)
    local_json = os.path.join(timing_dir, "producer.json")
    with open(local_json, "w") as fp:
        json.dump({
            "component":   "producer",
            "start_utc":   _START.isoformat() + "Z",
            "end_utc":     datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round(
                (datetime.datetime.utcnow() - _START).total_seconds(), 3
            )
        }, fp, indent=2)
    with open(local_json, "rb") as fp:
        save_bytes(f"{RESULT_DIR}/timing/producer.json",
                   fp.read(), "application/json")
    sync_all_metrics_to_minio()

    # KFP metadata 占位
    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
    print("[producer] ALL done.")

if __name__ == "__main__":
    main()
