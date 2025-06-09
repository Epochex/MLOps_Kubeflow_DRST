#!/usr/bin/env python3
"""
kafka_streaming/producer.py
────────────────────────────────────────────────────────────
Phase-1 / 2 / 3 依次读取 CSV → 发送 Kafka
★ 新增：所有阶段发送完毕后，发送一条 EOS（producer_done=True）。
"""
import os, sys, time, json, datetime
import pandas as pd, numpy as np
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.minio_helper import s3, BUCKET, load_csv
from shared.config       import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    TARGET_COL, BATCH_SIZE
)
from shared.features     import FEATURE_COLS
from shared.metric_logger import log_metric

SLEEP = float(os.getenv("MSG_SLEEP", "0.1"))
LIM1  = int(os.getenv("LIMIT_PHASE1", "3000"))
LIM2  = int(os.getenv("LIMIT_PHASE2", "800"))
LIM3  = int(os.getenv("LIMIT_PHASE3", "1000"))

STAGES = [
    ("Phase-1", f"{DATA_DIR}/random_rates.csv",                         LIM1),
    ("Phase-2", f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv", LIM2),
    ("Phase-3", f"{DATA_DIR}/intervention_global.csv",                 LIM3),
]

_START = datetime.datetime.utcnow()

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

def send_df(df: pd.DataFrame, producer: KafkaProducer, phase: str):
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

    prod = KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=lambda m: json.dumps(m).encode(),
    )

    for phase, rel_path, limit in STAGES:
        csv_path = _ensure_local(rel_path)
        df = (pd.read_csv(csv_path, index_col=0)
                .replace({'<not counted>': np.nan})
                .dropna()
                .reset_index(drop=True))
        if limit:
            df = df.iloc[:limit]
        send_df(df, prod, phase)

    # —— 发送终止哨兵 —— ------------------------------------
    prod.send(KAFKA_TOPIC, {
        "producer_done": True,
        "send_ts": datetime.datetime.utcnow().isoformat() + "Z"
    })
    prod.flush()
    prod.close()

    # —— 时序记录（保持原来逻辑） —— -------------------------
    timing_dir = os.path.join("results", "timing")
    os.makedirs(timing_dir, exist_ok=True)
    with open(os.path.join(timing_dir, "producer.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "component": "producer",
            "start_utc": _START.isoformat() + "Z",
            "end_utc":   datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round(
                (datetime.datetime.utcnow() - _START).total_seconds(), 3)
        }, fp, ensure_ascii=False, indent=2)

    print("[producer] ALL done.")

import json, os
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json","w") as f:
    json.dump({}, f)

if __name__ == "__main__":
    main()
