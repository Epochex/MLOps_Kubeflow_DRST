#!/usr/bin/env python3
"""
kafka_streaming/producer.py
────────────────────────────────────────────────────────────
Phase-1  : old_total.csv  → 与离线模型同分布，几乎 0 JS
Phase-2  : old_dag-1.csv → 触发 Drift & Retrain
Phase-3  : old_dag-1.csv → Monitor 重训完毕，Inference 热加载
"""
import os, sys, time, json, glob, datetime
import pandas as pd, numpy as np
from kafka import KafkaProducer

# 把项目根目录放进 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.minio_helper import s3, BUCKET, load_csv, save_bytes
from shared.config       import (
    DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
    TARGET_COL, BATCH_SIZE, RESULT_DIR
)
from shared.features      import FEATURE_COLS
from shared.metric_logger import log_metric, sync_all_metrics_to_minio

# ---------- 发送节奏 ----------
SLEEP = float(os.getenv("MSG_SLEEP", "0.1"))
LIM1  = int(os.getenv("LIMIT_PHASE1", "500"))
LIM2  = int(os.getenv("LIMIT_PHASE2", "1000"))
LIM3  = int(os.getenv("LIMIT_PHASE3", "1000"))

STAGES = [
    ("Phase-1", f"{DATA_DIR}/old_total.csv", LIM1),
    ("Phase-2", f"{DATA_DIR}/old_dag-1.csv", LIM2),
    ("Phase-3", f"{DATA_DIR}/old_dag-1.csv", LIM3),
]

_START = datetime.datetime.utcnow()

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
    加随机 key = os.urandom(4)，确保 round-robin 分区，
    三个 consumer 负载均衡。
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
        # ★ 关键改动：带随机 key
        prod.send(KAFKA_TOPIC,
                  key=os.urandom(4),
                  value=payload)

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

    # —— 等消费者 ——  
    num_consumers = int(os.getenv("NUM_CONSUMERS", "3"))
    timeout       = int(os.getenv("CONSUMER_WAIT_TIMEOUT", "60"))
    flag_glob     = f"/mnt/pvc/{RESULT_DIR}/consumer_ready_*.flag"
    t0 = time.time()
    print(f"[producer] waiting {num_consumers} consumers …")
    while time.time() - t0 < timeout:
        if len(glob.glob(flag_glob)) >= num_consumers:
            break
        time.sleep(1)

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
    for _ in range(num_consumers):
        prod.send(KAFKA_TOPIC,
                  key=os.urandom(4),   # ★ 同样带 key，避免全部去同一分区
                  value=sentinel)
    prod.flush(); prod.close()
    open(f"/mnt/pvc/{RESULT_DIR}/producer_done.flag", "w").close()
    print(f"[producer] sent {num_consumers} producer_done signals")

    # —— timing & metrics ——  
    timing_dir = os.path.join("/mnt/pvc", RESULT_DIR, "timing")
    os.makedirs(timing_dir, exist_ok=True)
    with open(os.path.join(timing_dir, "producer.json"), "w") as fp:
        json.dump({
            "component":   "producer",
            "start_utc":   _START.isoformat() + "Z",
            "end_utc":     datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round((datetime.datetime.utcnow() - _START).total_seconds(), 3)
        }, fp, indent=2)
    with open(os.path.join(timing_dir, "producer.json"), "rb") as fp:
        save_bytes(f"{RESULT_DIR}/timing/producer.json", fp.read(), "application/json")
    sync_all_metrics_to_minio()

    # —— KFP metadata 占位 ——  
    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")

    print("[producer] ALL done.")

if __name__ == "__main__":
    main()
