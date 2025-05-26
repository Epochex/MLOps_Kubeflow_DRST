# kafka_streaming/producer.py
#!/usr/bin/env python3
import json
import time
import pathlib
import sys
import os

import pandas as pd
import numpy as np
from kafka import KafkaProducer

# 如果你把 shared 放在 /app/shared 下，确保能导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.minio_helper import load_csv
from shared.features import FEATURE_COLS
from shared.config import DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS, TARGET_COL, BATCH_SIZE

def run_producer():
    # 1) 读全量数据（含缺失值处理）
    csv_path = os.path.join(DATA_DIR, "dag-1", "resource_stimulus_global_A-B-C.csv")
    df = load_csv(csv_path).reset_index(drop=True)
    total = len(df)
    print(f"[producer] total rows = {total}")

    # 2) 建立 Kafka 生产者
    producer = KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=lambda m: json.dumps(m).encode("utf-8")
    )

    # 3) 分批发送
    sent = 0
    for idx, row in df.iterrows():
        msg = {
            "row_index": int(idx),
            "features": {c: float(row[c]) for c in FEATURE_COLS},
            "label": float(row[TARGET_COL])
        }
        producer.send(KAFKA_TOPIC, msg)
        sent += 1

        # 每 BATCH_SIZE 条 flush 一次
        if sent % BATCH_SIZE == 0 or sent == total:
            producer.flush()
            batch_id = sent // BATCH_SIZE
            print(f"[producer] batch {batch_id:3d} | sent {sent}/{total}")

        # 控制一下频率
        time.sleep(0.05)

    producer.close()
    print("[producer] DONE")

if __name__ == "__main__":
    run_producer()

