#!/usr/bin/env python3
from __future__ import annotations
import os, time, json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd

from drst_common.config import (
    KAFKA_TOPIC, DATA_DIR, TARGET_COL, BUCKET,
    PRODUCE_INTERVAL_MS, PRODUCER_STAGES,
)
from drst_common.kafka_io import create_producer, broadcast_sentinel, create_consumer
from drst_common.minio_helper import s3
from drst_common.artefacts import load_selected_feats
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata

TOPIC        = os.getenv("KAFKA_TOPIC", KAFKA_TOPIC)
POD_NAME     = os.getenv("HOSTNAME", "producer")
MANIFEST_KEY = os.getenv("MANIFEST_KEY", f"{DATA_DIR}/online/manifest.json")

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _load_df_from_key(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

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
    if take == "tail":
        return df.tail(n)
    if take == "random":
        return df.sample(n=n, random_state=seed)
    return df.head(n)

def _iter_rows(df: pd.DataFrame, feats: List[str]):
    numdf = df.copy()
    for c in feats + ([TARGET_COL] if TARGET_COL in numdf.columns else []):
        if c not in numdf.columns:
            numdf[c] = 0.0
        numdf[c] = pd.to_numeric(numdf[c], errors="coerce").fillna(0.0)
    for _, r in numdf.iterrows():
        yield {
            "send_ts": _ts(),
            "features": {c: float(r[c]) for c in feats},
            "label": float(r[TARGET_COL]) if TARGET_COL in numdf.columns else None,
        }

def main():
    feats = load_selected_feats()
    prod  = create_producer()
    touch_ready("producer", POD_NAME)

    stages_manifest = _try_load_manifest_stages()
    stages = _normalize_stages(stages_manifest if stages_manifest else PRODUCER_STAGES)
    if not stages:
        raise RuntimeError("no producer stages defined (empty PRODUCER_STAGES and manifest missing)")

    total = 0
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
        for j, msg in enumerate(_iter_rows(df_sub, feats), 1):
            prod.send(TOPIC, msg)
            total += 1
            if PRODUCE_INTERVAL_MS > 0:
                time.sleep(PRODUCE_INTERVAL_MS / 1000.0)
        log_metric(component="producer", event=f"stage{idx}_done", value=n, kafka_topic=TOPIC)

    cons = create_consumer(TOPIC, group_id="cg-producer-probe")
    time.sleep(1.0)
    parts = cons.partitions_for_topic(TOPIC) or []
    broadcast_sentinel(prod, TOPIC, payload={"producer_done": True}, partitions=parts)

    log_metric(component="producer", event="total_sent", value=total, kafka_topic=TOPIC)
    sync_all_metrics_to_minio()
    write_kfp_metadata()
    print(f"[producer:{POD_NAME}] done. sent={total}, sentinels={len(parts)}")

if __name__ == "__main__":
    main()
