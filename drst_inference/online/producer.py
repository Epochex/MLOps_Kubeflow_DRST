#!/usr/bin/env python3
# drst_inference/online/producer.py
from __future__ import annotations
import time, json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd

from drst_common.config import (
    KAFKA_TOPIC, DATA_DIR, TARGET_COL, BUCKET,
    PRODUCE_INTERVAL_MS, PRODUCER_STAGES, BATCH_SIZE,
    PRODUCER_PARTITION_MODE, MANIFEST_KEY, MODEL_DIR
)
from drst_common.kafka_io import create_producer, broadcast_sentinel, create_consumer
from drst_common.minio_helper import s3
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.resource_probe import start as start_probe

TOPIC    = KAFKA_TOPIC
POD_NAME = "producer"

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _load_df_from_key(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

def _load_feature_cols(timeout_s: int = 120, interval_s: float = 2.0) -> List[str]:
    """等待 offline 产出的 feature_cols.json"""
    end = time.time() + timeout_s
    last_err = None
    key = f"{MODEL_DIR}/feature_cols.json"
    while time.time() < end:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            cols = json.loads(obj["Body"].read().decode())
            if isinstance(cols, list) and cols:
                return [str(c) for c in cols]
        except Exception as e:
            last_err = e
            time.sleep(interval_s)
    raise RuntimeError(f"wait feature_cols.json timeout: {last_err}")

def _try_load_manifest_stages() -> List[Tuple[str, int]] | None:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)
        m = json.loads(obj["Body"].read().decode())
        if isinstance(m.get("stages"), list):
            out: List[Tuple[str, int]] = []
            for it in m["stages"]:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    out.append((str(it[0]), int(it[1])))
                elif isinstance(it, dict) and "key" in it and "rows" in it:
                    out.append((str(it["key"]), int(it["rows"])))
            return out or None
    except Exception:
        pass
    return None

def _normalize_stages(stages) -> List[Dict]:
    out: List[Dict] = []
    for it in stages or []:
        if isinstance(it, dict):
            key = str(it.get("key") or "")
            rows = int(it.get("rows", 0))
            take = str(it.get("take", "head")).lower()
            seed = it.get("seed", None)
            if key and rows > 0:
                out.append({"key": key, "rows": rows, "take": take, "seed": seed})
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            out.append({"key": str(it[0]), "rows": int(it[1]), "take": "head", "seed": None})
    return out

def _subset_df(df: pd.DataFrame, rows: int, take: str, seed: Optional[int]):
    n = min(int(rows), len(df))
    if take == "tail":   return df.tail(n).reset_index(drop=True)
    if take == "random": return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)

def _iter_rows(df: pd.DataFrame, feat_all: List[str]):
    numdf = df.copy()
    # 只保证 FEATURE_COLS 存在且为数值；目标列不放入 features
    for c in feat_all:
        if c not in numdf.columns: numdf[c] = 0.0
        numdf[c] = pd.to_numeric(numdf[c], errors="coerce").fillna(0.0)
    for _, r in numdf.iterrows():
        yield {
            "send_ts": _ts(),
            "features": {c: float(r[c]) for c in feat_all},
            "label": float(r[TARGET_COL]) if TARGET_COL in numdf.columns else None,
        }

def main():
    stop_probe = start_probe("producer")
    try:
        feat_all = _load_feature_cols()  # 60 维 FEATURE_COLS
        prod  = create_producer()
        touch_ready("producer", POD_NAME)

        # 分区信息（用于严格 RR/Hash）
        parts = sorted(list(prod.partitions_for(TOPIC) or []))
        t0 = time.time()
        while not parts and (time.time() - t0) < 15:
            time.sleep(0.5); parts = sorted(list(prod.partitions_for(TOPIC) or []))
        if parts:
            print(f"[producer:{POD_NAME}] partitions discovered: {parts}")
        else:
            print(f"[producer:{POD_NAME}] partitions not visible; fallback to client default")

        rr_idx = 0
        part_count: Dict[int, int] = {p: 0 for p in parts}

        stages_manifest = _try_load_manifest_stages()
        stages = _normalize_stages(stages_manifest if stages_manifest else PRODUCER_STAGES)
        if not stages:
            raise RuntimeError("no producer stages defined (empty PRODUCER_STAGES and manifest missing)")

        total = 0
        sent_in_batch = 0  # 用于按 BATCH_SIZE 节拍休眠

        for idx, spec in enumerate(stages, 1):
            key, rows, take, seed = spec["key"], spec["rows"], spec.get("take", "head"), spec.get("seed", None)
            try:
                df = _load_df_from_key(key)
            except Exception as e:
                print(f"[producer:{POD_NAME}] fail {key}: {e} (skip stage {idx})")
                continue

            df_sub = _subset_df(df, rows, take, seed)
            n = len(df_sub)
            print(f"[producer:{POD_NAME}] stage{idx}: key={key} take={take} rows={n}")

            for _, msg in enumerate(_iter_rows(df_sub, feat_all), 1):
                # 发送
                if PRODUCER_PARTITION_MODE == "rr" and parts:
                    p = parts[rr_idx % len(parts)]
                    prod.send(TOPIC, msg, partition=p)
                    part_count[p] = part_count.get(p, 0) + 1
                    rr_idx += 1
                elif PRODUCER_PARTITION_MODE == "hash" and parts:
                    prod.send(TOPIC, msg, key=str(rr_idx).encode("utf-8"))
                    rr_idx += 1
                else:
                    prod.send(TOPIC, msg)

                total += 1
                sent_in_batch += 1

                # —— 节拍：每 BATCH_SIZE 条后再 sleep —— #
                if PRODUCE_INTERVAL_MS > 0 and sent_in_batch >= max(1, BATCH_SIZE):
                    time.sleep(PRODUCE_INTERVAL_MS / 1000.0)
                    sent_in_batch = 0

            # 每个 stage 结束打点
            log_metric(component="producer", event=f"stage{idx}_done", value=n, kafka_topic=TOPIC)

        # 发送 sentinel（每个分区一个）
        cons = create_consumer(TOPIC, group_id="cg-producer-probe")
        time.sleep(1.0)
        parts_for_sentinel = cons.partitions_for_topic(TOPIC) or []
        broadcast_sentinel(create_producer(), TOPIC, payload={"producer_done": True}, partitions=parts_for_sentinel)

        # 分区分布统计
        if part_count:
            for p, c in sorted(part_count.items()):
                log_metric(component="producer", event="partition_count", partition=int(p), value=int(c))
            print(f"[producer:{POD_NAME}] partition counts: {part_count}")

        log_metric(component="producer", event="total_sent", value=total, kafka_topic=TOPIC)
        sync_all_metrics_to_minio()
        write_kfp_metadata()
        print(f"[producer:{POD_NAME}] done. sent={total}, sentinels={len(parts_for_sentinel)}")
    finally:
        stop_probe()

if __name__ == "__main__":
    main()
