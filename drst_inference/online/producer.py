#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, random
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

from drst_common.config import (
    BUCKET, MODEL_DIR, TARGET_COL, DATA_DIR,
    PRODUCER_BRIDGE_N, PRODUCER_RAND_N, PRODUCER_STIM_N,
    PRODUCER_TPS, PRODUCER_JITTER_MS, PRODUCE_INTERVAL_MS
)
from drst_common.minio_helper import load_csv, s3
from drst_common.kafka_io import create_producer, broadcast_sentinel, partitions_for_topic
from drst_common.resource_probe import start as start_probe  # <<< 新增

TOPIC = os.getenv("KAFKA_TOPIC", "latencyTopic")
RUN_ID = os.getenv("RUN_ID") or time.strftime("%Y%m%d%H%M%S", time.gmtime())

BRIDGE_N = int(os.getenv("BRIDGE_N", str(PRODUCER_BRIDGE_N)))
RAND_N   = int(os.getenv("RAND_N",   str(PRODUCER_RAND_N)))
STIM_N   = int(os.getenv("STIM_N",   str(PRODUCER_STIM_N)))

KEY_COMBINED = os.getenv("STAGE1_KEY", f"{DATA_DIR}/combined.csv")
KEY_RANDOM   = os.getenv("STAGE2_KEY", f"{DATA_DIR}/random_rates.csv")
KEY_STIM     = os.getenv("STAGE3_KEY", f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv")

WAIT_FEATURES_SECS = int(os.getenv("WAIT_FEATURES_SECS", "120"))

def _per_msg_sleep_s() -> float:
    tps_env = os.getenv("PRODUCER_TPS")
    if tps_env is not None and tps_env.strip():
        try:
            tps = float(tps_env)
            if tps > 0:
                return 1.0 / tps
        except Exception:
            pass
    if PRODUCER_TPS and PRODUCER_TPS > 0:
        return 1.0 / float(PRODUCER_TPS)
    interval_ms_env = os.getenv("INTERVAL_MS")
    if interval_ms_env is not None and interval_ms_env.strip():
        try:
            ms = float(interval_ms_env)
            if ms >= 0:
                return ms / 1000.0
        except Exception:
            pass
    return float(PRODUCE_INTERVAL_MS) / 1000.0

def _sleep_for_rate(base_s: float):
    j_ms_env = os.getenv("PRODUCER_JITTER_MS")
    jitter_ms = int(j_ms_env) if j_ms_env else int(PRODUCER_JITTER_MS)
    if jitter_ms > 0:
        time.sleep(base_s + random.randint(0, jitter_ms) / 1000.0)
    else:
        time.sleep(base_s)

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
    # 资源采样（producer）
    stop_probe = start_probe("producer")

    try:
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

        per_msg = _per_msg_sleep_s()
        if per_msg > 0:
            tps = 1.0 / per_msg if per_msg > 0 else 0.0
            print(f"[producer:producer] throttling enabled: {tps:.2f} msgs/sec (~{per_msg*1000:.0f} ms/record)", flush=True)

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
                _sleep_for_rate(per_msg)

        n_sentinels = broadcast_sentinel(producer, TOPIC, run_id=RUN_ID, partitions=parts)
        try:
            producer.flush(10)
            producer.close(10)
        except Exception:
            pass

        print(f"[producer:producer] done. sent={sent}, sentinels={n_sentinels}", flush=True)

    finally:
        # 强制 flush + 写运行时长
        stop_probe()

if __name__ == "__main__":
    main()
