#!/usr/bin/env python3
"""
kafka_streaming/producer.py
────────────────────────────────────────────────────────────
Phase-1 / 2 / 3 依次读取 CSV → 发送 Kafka

★ 主要特性
1. 若 <repo 根>/<DATA_DIR>/… 文件不存在，
   则从 MinIO 下载并写到本地；
2. 默认流速 0.1 s/msg，可用环境变量 MSG_SLEEP 覆盖；
3. 发送完每个阶段，将行数等信息写入 metrics_summary.csv；
4. 【新增】给每条消息加上 send_ts，供 RTT 计算。
"""
import os
import sys
import time
import json
import datetime
import pandas as pd
import numpy as np
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.minio_helper import s3, BUCKET, load_csv
from shared.config       import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    TARGET_COL, BATCH_SIZE
)
from shared.features     import FEATURE_COLS
from shared.metric_logger import log_metric

# ─────────── 可调参数 ───────────
SLEEP = float(os.getenv("MSG_SLEEP", "0.1"))  # 100 ms / msg
LIM1  = int(os.getenv("LIMIT_PHASE1", "3000"))
LIM2  = int(os.getenv("LIMIT_PHASE2", "800"))
LIM3  = int(os.getenv("LIMIT_PHASE3", "1000"))

STAGES = [
    ("Phase-1", f"{DATA_DIR}/random_rates.csv",                        LIM1),
    ("Phase-2", f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv", LIM2),
    ("Phase-3", f"{DATA_DIR}/intervention_global.csv",                LIM3),
]

_START = datetime.datetime.utcnow()

def _ensure_local(rel_path: str) -> str:
    """
    返回本地文件绝对路径；若文件不存在，则从 MinIO 同名 Key
    下载并写到磁盘。
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
    把 DataFrame 逐行发送到 Kafka，并在每条消息中加上 send_ts。
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

        # 每 BATCH_SIZE 条 flush 一次，并打印进度
        if sent % BATCH_SIZE == 0 or sent == total:
            producer.flush()
            batch_id += 1
            print(f"[producer] {phase} | batch {batch_id:03d} | {sent}/{total}")

        time.sleep(SLEEP)

    # 阶段完成埋点
    log_metric(
        component="producer",
        event="stage_done",
        phase=phase,
        rows=int(total)
    )
    print(f"[producer] ✓ finished {phase}")

def main():
    print("[producer] start …")

    prod = KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=lambda m: json.dumps(m).encode(),
    )

    # 依次按阶段读取、截断、发送
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

    prod.close()

    # ——— 时序记录 ———
    timing_dir = os.path.join("results", "timing")
    os.makedirs(timing_dir, exist_ok=True)
    with open(os.path.join(timing_dir, "producer.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "component": "producer",
            "start_utc": _START.isoformat() + "Z",
            "end_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round((datetime.datetime.utcnow() - _START).total_seconds(), 3)
        }, fp, ensure_ascii=False, indent=2)

    print("[producer] ALL done.")

if __name__ == "__main__":
    main()
