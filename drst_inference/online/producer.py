# DRST-SoftwarizedNetworks/drst_inference/online/producer.py
#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd

from drst_common.config import (
    BUCKET, MODEL_DIR, TARGET_COL, DATA_DIR,
    PRODUCER_STAGES, PRODUCER_TPS, PRODUCER_JITTER_MS, PRODUCE_INTERVAL_MS,
    WAIT_FEATURES_SECS,
)
from drst_common.minio_helper import load_csv, s3
from drst_common.kafka_io import create_producer, broadcast_sentinel, partitions_for_topic
from drst_common.resource_probe import start as start_probe

# ------- 基本参数 -------
TOPIC  = os.getenv("KAFKA_TOPIC", "latencyTopic")
RUN_ID = os.getenv("RUN_ID") or time.strftime("%Y%m%d%H%M%S", time.gmtime())

# 白名单（默认使用 config 里定义的顺序与名称）
# 例如：PRODUCER_STAGES="stage0,stage1,stage2,stage3"
STAGE_WHITELIST = {s.strip().lower() for s in os.getenv("PRODUCER_STAGES", "").split(",") if s.strip()}

# ------- 发送节流 -------
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
    jitter_ms = int(j_ms_env) if (j_ms_env and j_ms_env.strip()) else int(PRODUCER_JITTER_MS)
    if jitter_ms > 0:
        time.sleep(base_s + random.randint(0, jitter_ms) / 1000.0)
    else:
        time.sleep(base_s)

# ------- 等待并读取特征列 -------
def _load_feature_cols() -> List[str]:
    key = f"{MODEL_DIR}/feature_cols.json"
    deadline = time.time() + int(WAIT_FEATURES_SECS)
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

# ------- 数据裁剪/对齐 -------
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

def _select_rows(df: pd.DataFrame, how: str, n: int) -> pd.DataFrame:
    if n is None or int(n) <= 0:
        return df.copy()
    n = int(n)
    if how == "tail":
        return df.tail(n)
    return df.head(n)

# ------- 阶段定义 -------
@dataclass
class StageDef:
    name: str
    key:  str
    take: str
    rows: int

def _load_stage_defs() -> List[StageDef]:
    defs: List[StageDef] = []
    for idx, d in enumerate(PRODUCER_STAGES):
        name = str(d.get("name") or f"stage{idx}")
        key  = str(d.get("key"))
        take = str(d.get("take", "head")).lower()
        rows = int(d.get("rows", 0))
        defs.append(StageDef(name=name, key=key, take=take, rows=rows))
    return defs

def _load_stage_frames(feat_cols: List[str]) -> List[Tuple[StageDef, pd.DataFrame]]:
    pairs: List[Tuple[StageDef, pd.DataFrame]] = []
    for sd in _load_stage_defs():
        if STAGE_WHITELIST and (sd.name.lower() not in STAGE_WHITELIST):
            print(f"[producer] skip stage={sd.name} (PRODUCER_STAGES whitelist)", flush=True)
            continue
        try:
            df = load_csv(sd.key)
            df = _select_rows(df, sd.take, sd.rows)
            df = _align(df, feat_cols)
            print(f"[producer] {sd.name:<7} key={sd.key} take={sd.take:<4} rows={len(df)}", flush=True)
        except Exception as e:
            print(f"[producer] {sd.name} load failed: {e}; use empty.", flush=True)
            df = pd.DataFrame(columns=feat_cols + [TARGET_COL])
        pairs.append((sd, df))
    return pairs

# ------- 主流程 -------
def main():
    stop_probe = start_probe("producer")
    try:
        feat_cols = _load_feature_cols()
        parts = partitions_for_topic(TOPIC)
        print(f"[producer] topic={TOPIC} partitions={parts}", flush=True)

        frames = _load_stage_frames(feat_cols)
        per_msg = _per_msg_sleep_s()
        if per_msg > 0:
            print(f"[producer] throttling: ~{(1.0/per_msg):.2f} msg/s (≈{per_msg*1000:.0f} ms/record)", flush=True)

        producer = create_producer(client_id=f"producer-{RUN_ID}")
        sent = 0

        for sd, df in frames:
            if df.empty:
                print(f"[producer] stage={sd.name} is empty; skip.", flush=True)
                continue
            for _, row in df.iterrows():
                payload = {
                    "features": {c: float(row[c]) for c in feat_cols},
                    "label":    float(row[TARGET_COL]),
                    "send_ts":  datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                    "stage":    sd.name,
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
        print(f"[producer] done. sent={sent}, sentinels={n_sentinels}", flush=True)

    finally:
        stop_probe()

if __name__ == "__main__":
    main()
