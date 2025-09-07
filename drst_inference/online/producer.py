#!/usr/bin/env python3
# drst_inference/online/producer.py
from __future__ import annotations
import os, json, time, itertools, hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from drst_common.config import (
    BUCKET, MODEL_DIR, TARGET_COL,
    PRODUCE_INTERVAL_MS, PRODUCER_STAGES, PRODUCER_PARTITION_MODE, KAFKA_TOPIC
)
from drst_common.minio_helper import load_csv, s3
from drst_common.kafka_io import create_producer, broadcast_sentinel, partitions_for_topic

RUN_ID = os.getenv("RUN_ID") or time.strftime("%Y%m%d%H%M%S", time.gmtime())

WAIT_FEATURES_SECS = int(os.getenv("WAIT_FEATURES_SECS", "120"))

def _sleep_ms(ms: int):
    if ms > 0:
        time.sleep(ms / 1000.0)

def _load_feature_cols() -> List[str]:
    key = f"{MODEL_DIR}/feature_cols.json"
    deadline = time.time() + WAIT_FEATURES_SECS
    last_err = None
    while time.time() < deadline:
        try:
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            cols = json.loads(raw.decode("utf-8"))
            if not isinstance(cols, list) or not cols:
                raise ValueError("feature_cols.json invalid")
            return cols
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"wait feature_cols.json timeout: {last_err}")

def _select_rows(df: pd.DataFrame, how: str, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    if how == "tail":
        return df.tail(n)
    return df.head(n)

def _align(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # 对齐列：仅取 feature_cols + TARGET_COL，多余列丢弃，缺失补 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = np.nan
    keep = feature_cols + [TARGET_COL]
    df = df[keep].copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df

def _stages_from_env_or_config() -> List[dict]:
    # 若设置了 PRODUCER_STAGES(JSON)，优先用；否则用 config.PRODUCER_STAGES
    raw = os.getenv("PRODUCER_STAGES", "").strip()
    if raw:
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                return arr
        except Exception:
            pass
    return list(PRODUCER_STAGES)

def _choose_partition_iter(mode: str, parts: List[int]):
    """返回一个迭代器，依 mode 选择分区。"""
    if not parts:
        while True:
            yield None
    mode = (mode or "auto").lower()
    if mode == "rr":
        cyc = itertools.cycle(parts)
        while True:
            yield next(cyc)
    elif mode == "hash":
        i = 0; n = len(parts)
        while True:
            # 简单 hash：按计数对分区数取模
            yield parts[i % n]; i += 1
    else:  # "auto"
        while True:
            yield None

def main():
    topic = os.getenv("KAFKA_TOPIC", KAFKA_TOPIC)
    interval_ms = int(os.getenv("PRODUCE_INTERVAL_MS", str(PRODUCE_INTERVAL_MS)))
    stages = _stages_from_env_or_config()
    feat_cols = _load_feature_cols()

    # 发现分区
    parts = partitions_for_topic(topic)
    print(f"[producer:producer] partitions discovered: {parts}", flush=True)

    # 读数据并对齐
    dfs: List[pd.DataFrame] = []
    for st in stages:
        key  = str(st.get("key"))
        take = str(st.get("take", "head")).lower()
        rows = int(st.get("rows", 0))
        df   = _align(_select_rows(load_csv(key), take, rows), feat_cols)
        print(f"[producer:producer] stage: key={key} take={take} rows={len(df)}", flush=True)
        if not df.empty:
            dfs.append(df)

    # Producer
    producer = create_producer(client_id=f"producer-{RUN_ID}")
    sent = 0
    per_partition: Dict[int, int] = {p: 0 for p in parts}
    pick_part = _choose_partition_iter(os.getenv("PRODUCER_PARTITION_MODE", PRODUCER_PARTITION_MODE), parts)

    for df in dfs:
        for _, row in df.iterrows():
            # 与 inference_consumer 约定的消息结构
            payload = {
                "features": {c: float(row[c]) for c in feat_cols},
                "label": float(row[TARGET_COL]),
                "send_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": RUN_ID,
            }
            p = next(pick_part)
            fut = producer.send(topic, value=payload,
                                partition=p,
                                headers=[("run_id", RUN_ID.encode("utf-8"))])
            # 统计分区
            try:
                md = fut.get(timeout=5)
                if md and md.partition is not None:
                    per_partition[int(md.partition)] = per_partition.get(int(md.partition), 0) + 1
            except Exception:
                pass
            sent += 1
            _sleep_ms(interval_ms)

    # 广播哨兵（每个分区 1 个，带 run_id）
    n_sentinels = broadcast_sentinel(producer, topic, run_id=RUN_ID, partitions=parts)
    try:
        producer.flush(10)
        producer.close(10)
    except Exception:
        pass

    print(f"[producer:producer] partition counts: {per_partition}", flush=True)
    print(f"[producer:producer] done. sent={sent}, sentinels={n_sentinels}", flush=True)

if __name__ == "__main__":
    main()
