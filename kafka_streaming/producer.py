#!/usr/bin/env python3
"""
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
SLEEP = float(os.getenv("MSG_SLEEP", "0.5"))
LIM1  = int(os.getenv("LIMIT_PHASE1", "1000"))
LIM2  = int(os.getenv("LIMIT_PHASE2", "700"))
LIM3  = int(os.getenv("LIMIT_PHASE3", "250"))


STAGES = [
    ("Phase-1", f"datasets/random_rates.csv", LIM1),
    ("Phase-2", f"datasets/resource_stimulus_global_A-B-C_modified.csv", LIM2),
    ("Phase-3", f"datasets/resource_stimulus_global_A-B-C_modified.csv", LIM3),
]

# ---------- 本地缓存 & 时间戳 ----------
TMP_DIR = "/tmp/producer"             # ← 全面改用 /tmp
os.makedirs(TMP_DIR, exist_ok=True)

_START = datetime.datetime.utcnow()   # 保留原来的启动时间

# producer_done 本地标记（只给本进程参考）
FLAG_DONE_LOCAL = os.path.join(TMP_DIR, "producer_done.flag")


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
        prod.send(KAFKA_TOPIC, value=payload)  # ✅ 让 RoundRobinPartitioner 接管


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
    time.sleep(5)  # 等待其他组件准备就绪
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

    # —— 等consumer 在 MinIO 上写好就绪标记 ——  
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
        # 删除 partitioner 参数即可，默认就是轮询策略
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

    # 获取 topic 的所有分区，并将 sentinel 分发到每个分区
    partitions = prod.partitions_for(KAFKA_TOPIC)
    for p in partitions:
        prod.send(
            KAFKA_TOPIC,
            partition=p,
            value=sentinel
        )

    prod.flush()
    prod.close()

    # ① 本地 /tmp 写一个标记文件（可选，仅供诊断）
    open(FLAG_DONE_LOCAL, "w").close()

    # ② 上传 MinIO，让 Monitor / Inference 通过 MinIO 同步
    save_bytes(f"{RESULT_DIR}/producer_done.flag", b"", "text/plain")
    print(f"[producer] sent sentinel to {len(partitions)} partitions")

    # —— timing & metrics ——  
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

    # 上传 timing 到 MinIO
    with open(local_json, "rb") as fp:
        save_bytes(f"{RESULT_DIR}/timing/producer.json",
                   fp.read(), "application/json")

    sync_all_metrics_to_minio()

    # —— KFP metadata 占位 ——  
    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")

    print("[producer] ALL done.")


if __name__ == "__main__":
    main()
