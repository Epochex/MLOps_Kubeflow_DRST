#!/usr/bin/env python3
# drst_inference/online/producer.py
from __future__ import annotations
import os, json, time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

from drst_common.config import BUCKET, MODEL_DIR, TARGET_COL
from drst_common.minio_helper import load_csv, s3
from drst_common.kafka_io import create_producer, broadcast_sentinel, partitions_for_topic

TOPIC        = os.getenv("KAFKA_TOPIC", "latencyTopic")
INTERVAL_MS  = int(os.getenv("INTERVAL_MS", "100"))
RUN_ID       = os.getenv("RUN_ID") or time.strftime("%Y%m%d%H%M%S", time.gmtime())

BRIDGE_N     = int(os.getenv("BRIDGE_N", "500"))
RAND_N       = int(os.getenv("RAND_N", "1000"))
STIM_N       = int(os.getenv("STIM_N", "1000"))

KEY_COMBINED = os.getenv("STAGE1_KEY", "datasets/combined.csv")
KEY_RANDOM   = os.getenv("STAGE2_KEY", "datasets/random_rates.csv")
KEY_STIM     = os.getenv("STAGE3_KEY", "datasets/resource_stimulus_global_A-B-C_modified.csv")

WAIT_FEATURES_SECS = int(os.getenv("WAIT_FEATURES_SECS", "120"))

def _sleep_ms(ms: int): 
    if ms > 0: time.sleep(ms / 1000.0)

def _load_feature_cols() -> List[str]:
    key = f"{MODEL_DIR}/feature_cols.json"
    deadline = time.time() + WAIT_FEATURES_SECS
    last_err = None
    while time.time() < deadline:
        try:
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            cols = json.loads(raw.decode("utf-8"))
            if isinstance(cols, list) and cols:
                return [str(c) for c in cols]
        except Exception as e:
            last_err = e
        time.sleep(2)
    raise RuntimeError(f"wait feature_cols.json timeout: {last_err}")

def _select_rows(df: pd.DataFrame, how: str, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.tail(n) if how == "tail" else df.head(n)

def _align(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
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

def main():
    feat_cols = _load_feature_cols()
    parts = partitions_for_topic(TOPIC)
    print(f"[producer:producer] partitions discovered: {parts}", flush=True)

    df1 = _select_rows(load_csv(KEY_COMBINED), "tail", BRIDGE_N)
    print(f"[producer:producer] stage1: key={KEY_COMBINED} take=tail rows={len(df1)}", flush=True)
    df2 = _select_rows(load_csv(KEY_RANDOM), "head", RAND_N)
    print(f"[producer:producer] stage2: key={KEY_RANDOM} take=head rows={len(df2)}", flush=True)
    try:
        df3 = _select_rows(load_csv(KEY_STIM), "head", STIM_N)
    except Exception:
        df3 = pd.DataFrame()
    print(f"[producer:producer] stage3: key={KEY_STIM} take=head rows={len(df3)}", flush=True)

    df_list = [
        _align(df1, feat_cols),
        _align(df2, feat_cols),
        (_align(df3, feat_cols) if not df3.empty else pd.DataFrame(columns=feat_cols+[TARGET_COL]))
    ]

    producer = create_producer(client_id=f"producer-{RUN_ID}")
    sent = 0
    for df in df_list:
        if df.empty: continue
        for _, row in df.iterrows():
            payload = {
                "features": {c: float(row[c]) for c in feat_cols},
                "label":    float(row[TARGET_COL]),
                "send_ts":  datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            }
            producer.send(TOPIC, value=payload, headers=[("run_id", RUN_ID.encode("utf-8"))])
            sent += 1
            _sleep_ms(INTERVAL_MS)

    n_sentinels = broadcast_sentinel(producer, TOPIC, run_id=RUN_ID, partitions=parts)
    try:
        producer.flush(10)
        producer.close(10)
    except Exception:
        pass

    print(f"[producer:producer] done. sent={sent}, sentinels={n_sentinels}", flush=True)

if __name__ == "__main__":
    main()
