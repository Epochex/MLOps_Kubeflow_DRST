# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/registry.py
from __future__ import annotations
import io, json, time
from typing import Any, Dict
from drst_common.minio_helper import s3, BUCKET, save_bytes
from drst_common.config import MODEL_DIR

REGISTRY_KEY = f"{MODEL_DIR}/forecasting/registry.json"

def load_registry() -> Dict[str, Any] | None:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=REGISTRY_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

def save_registry(meta: Dict[str, Any]) -> None:
    meta = dict(meta)
    meta["updated_ts"] = int(time.time())
    save_bytes(REGISTRY_KEY, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
               "application/json")
