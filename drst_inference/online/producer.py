#!/usr/bin/env python3
# drst_inference/online/producer.py
from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

from drst_common.config import KAFKA_TOPIC, DATA_DIR, TARGET_COL
from drst_common.kafka_io import create_producer, broadcast_sentinel, create_consumer, partitions_count
from drst_common.minio_helper import s3
from drst_common.artefacts import load_selected_feats
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.config import BUCKET

# ---- 参数（可环境变量覆盖）----
INTERVAL_MS   = int(os.getenv("PRODUCE_INTERVAL_MS", "100"))  # 100ms
MAX_ROWS_S1   = int(os.getenv("STAGE1_N", "3000"))
MAX_ROWS_S2   = int(os.getenv("STAGE2_N", "1000"))
MAX_ROWS_S3   = int(os.getenv("STAGE3_N", "1000"))
TOPIC         = os.getenv("KAFKA_TOPIC", KAFKA_TOPIC)
MANIFEST_KEY  = os.getenv("MANIFEST_KEY", f"{DATA_DIR}/online/manifest.json")
POD_NAME      = os.getenv("HOSTNAME", "producer")

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _load_df_from_key(key: str) -> pd.DataFrame:
    # 轻量读取：把对象流进 pandas
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

def _load_manifest() -> Dict:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)
        import json
        return json.loads(obj["Body"].read().decode())
    except Exception as e:
        print(f"[producer:{POD_NAME}] failed to load manifest {MANIFEST_KEY}: {e}")
        return {}
