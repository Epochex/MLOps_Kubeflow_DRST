#!/usr/bin/env python3
"""
kafka_streaming/producer.py
────────────────────────────────────────────────────────────
• Phase-1 / 2 / 3 依次读取 CSV → 发送 Kafka
• 启动前等待所有消费者写入就绪标志
• 发送结束后将本地结果和指标一次性上传到 MinIO
"""
import os
import sys
import time
import json
import glob
import datetime

import pandas as pd
import numpy as np
from kafka import KafkaProducer

# 把项目根目录添加到 sys.path，以便 import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.minio_helper import s3, BUCKET, load_csv, save_bytes
from shared.config       import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    TARGET_COL, BATCH_SIZE, RESULT_DIR
)
from shared.features      import FEATURE_COLS
from shared.metric_logger import log_metric, sync_all_metrics_to_minio

SLEEP = float(os.getenv("MSG_SLEEP", "0.1"))
LIM1  = int(os.getenv("LIMIT_PHASE1", "600"))
LIM2  = int(os.getenv("LIMIT_PHASE2", "1000"))
LIM3  = int(os.getenv("LIMIT_PHASE3", "1000"))

STAGES = [
    ("Phase-1", f"{DATA_DIR}/random_rates.csv",                         LIM1),
    ("Phase-2", f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv", LIM2),
    ("Phase-3", f"{DATA_DIR}/intervention_global.csv",                 LIM3),
]

_START = datetime.datetime.utcnow()

def _ensure_local(rel_path: str) -> str:
    """
    确保 CSV 文件在本地存在，如不存在则从 MinIO 下载。
    返回本地绝对路径。
    """
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

def send_df(df: pd.DataFrame, producer: KafkaProducer, phase: str):
    """
    按行发送 DataFrame 到 Kafka，发送完成后打点并打印。
    """
    total, sent, batch_id = len(df), 0, 0
    for idx, row in df.iterrows():
        payload = {
            "phase": phase,
            "row_index": int(idx),
            "features": {c: float(row[c]) for c in FEATURE_COLS},
            "label": float(row[TARGET_COL]),
            "send_ts": datetime.datetime.utcnow().isoformat() + "Z"
        }
        producer.send(KAFKA_TOPIC, payload)
        sent += 1
        if sent % BATCH_SIZE == 0 or sent == total:
            producer.flush()
            batch_id += 1
            print(f"[producer] {phase} | batch {batch_id:03d} | {sent}/{total}")
        time.sleep(SLEEP)

    log_metric(component="producer", event="stage_done",
               phase=phase, rows=int(total))
    print(f"[producer] ✓ finished {phase}")

def main():
    print("[producer] start …")

    # —— 在真正发送前，等待所有消费者写就绪标志 ——  
    num_consumers = int(os.getenv("NUM_CONSUMERS", "1"))
    timeout       = int(os.getenv("CONSUMER_WAIT_TIMEOUT", "60"))
    flags_pattern = f"/mnt/pvc/{RESULT_DIR}/consumer_ready_*.flag"
    start_ts = time.time()
    print(f"[producer] waiting for {num_consumers} consumers to be ready (timeout {timeout}s)…")
    while time.time() - start_ts < timeout:
        flags = glob.glob(flags_pattern)
        if len(flags) >= num_consumers:
            print(f"[producer] detected {len(flags)} readiness flags, proceeding.")
            break
        time.sleep(1)
    else:
        print(f"[producer] timeout waiting for consumers, proceeding anyway.")

    # —— 初始化 KafkaProducer ——  
    prod = KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=lambda m: json.dumps(m).encode(),
    )

    # —— 分阶段读取 CSV 并发送 ——  
    for phase, rel_path, limit in STAGES:
        csv_path = _ensure_local(rel_path)
        df = (
            pd.read_csv(csv_path, index_col=0)
              .replace({'<not counted>': np.nan})
              .dropna()
              .reset_index(drop=True)
        )
        if limit:
            df = df.iloc[:limit]
        send_df(df, prod, phase)

    # —— 发送终止哨兵：每个 consumer 一条 ——  
    sentinel = {
        "producer_done": True,
        "send_ts": datetime.datetime.utcnow().isoformat() + "Z"
    }
    for _ in range(num_consumers):
        prod.send(KAFKA_TOPIC, sentinel)
    prod.flush()
    prod.close()
    open(f"/mnt/pvc/{RESULT_DIR}/producer_done.flag", "w").close()
    print(f"[producer] sent {num_consumers} producer_done signals and touched producer_done.flag")

    # —— 时序记录到 PVC 的结果目录 ——  
    timing_dir = os.path.join("/mnt/pvc", RESULT_DIR, "timing")
    os.makedirs(timing_dir, exist_ok=True)
    with open(os.path.join(timing_dir, "producer.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "component":   "producer",
            "start_utc":   _START.isoformat() + "Z",
            "end_utc":     datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round((datetime.datetime.utcnow() - _START).total_seconds(), 3)
        }, fp, ensure_ascii=False, indent=2)

    print("[producer] ALL done.")

    # —— 上传时序记录和所有本地指标到 MinIO ——  
    with open(os.path.join(timing_dir, "producer.json"), "rb") as fp:
        save_bytes(f"{RESULT_DIR}/timing/producer.json", fp.read(), "application/json")
    sync_all_metrics_to_minio()

    # —— 写 KFP V2 metadata.json ——  
    meta_dir = "/tmp/kfp_outputs"
    os.makedirs(meta_dir, exist_ok=True)
    with open(f"{meta_dir}/output_metadata.json", "w") as f:
        json.dump({}, f)

if __name__ == "__main__":
    main()
