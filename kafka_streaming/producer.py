#!/usr/bin/env python3
"""
kafka_streaming/producer.py
────────────────────────────────────────────────────────────
按 Phase 顺序把 CSV 推入 Kafka，并允许通过环境变量
快速限制每个 Phase 的行数，默认非常小，方便功能性自测。

  环境变量（可选）
  ──────────────────────────────────────────
  • LIMIT_PHASE1   (int)   默认 100
  • LIMIT_PHASE2   (int)   默认 200
  • LIMIT_PHASE3   (int)   默认 300
"""

import os, sys, time, datetime, json, pandas as pd
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.minio_helper import load_csv
from shared.config       import (
    DATA_DIR, RESULT_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    TARGET_COL, BATCH_SIZE
)
from shared.features     import FEATURE_COLS

# ─────────── 可调参数 ───────────
SLEEP_PER_MSG = float(os.getenv("MSG_SLEEP", "0.05"))  # 50 ms
LIM1 = int(os.getenv("LIMIT_PHASE1", "100"))
LIM2 = int(os.getenv("LIMIT_PHASE2", "200"))
LIM3 = int(os.getenv("LIMIT_PHASE3", "300"))
# LIM1 = int(os.getenv("LIMIT_PHASE1", "1000"))
# LIM2 = int(os.getenv("LIMIT_PHASE2", "2000"))
# LIM3 = int(os.getenv("LIMIT_PHASE3", "10000"))

_START = datetime.datetime.utcnow()

def send_df(df: pd.DataFrame, producer: KafkaProducer, phase: str):
    total, sent, batch_id = len(df), 0, 0
    for idx, row in df.iterrows():
        producer.send(KAFKA_TOPIC, {
            "phase": phase,
            "row_index": int(idx),
            "features": {c: float(row[c]) for c in FEATURE_COLS},
            "label": float(row[TARGET_COL]),
        })
        sent += 1
        if sent % BATCH_SIZE == 0 or sent == total:
            producer.flush()
            batch_id += 1
            print(f"[producer] {phase} | batch {batch_id:03d} | {sent}/{total}")
        time.sleep(SLEEP_PER_MSG)
    print(f"[producer] ✓ finished {phase}")

def main():
    prod = KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=lambda m: json.dumps(m).encode(),
    )

    stages = [
        ("Phase-1", f"{DATA_DIR}/random_rates.csv",                        LIM1),
        ("Phase-2", f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv", LIM2),
        ("Phase-3", f"{DATA_DIR}/intervention_global.csv",                LIM3),
    ]

    for phase, key, limit in stages:
        df = load_csv(key).reset_index(drop=True)
        if limit:       # 0 或 None 表示不截断
            df = df.iloc[:limit]
        send_df(df, prod, phase)

    prod.close()

    # 本地 timing
    timing_dir = os.path.join(RESULT_DIR, "timing")
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
