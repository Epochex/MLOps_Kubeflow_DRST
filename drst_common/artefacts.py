#!/usr/bin/env python3
# drst_common/artefacts.py
from __future__ import annotations
import io, os, json, time
from typing import List, Optional, Tuple, Dict

import joblib
from botocore.exceptions import ClientError

from .config import BUCKET, MODEL_DIR, RESULT_DIR
from .minio_helper import s3, save_bytes

# 等待离线工件就绪（避免刚写入就去读的竞态）
_WAIT_TOTAL_S   = int(os.getenv("OFFLINE_ARTIFACT_WAIT_S", "120"))
_WAIT_INTERVALS = [1, 2, 3, 4, 5]  # 渐进回退，之后固定 5 秒

def _head_ok(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    except Exception:
        return False

def _wait_keys(keys: List[str]) -> None:
    if _WAIT_TOTAL_S <= 0:
        return
    deadline = time.time() + _WAIT_TOTAL_S
    i = 0
    missing = set(keys)
    while missing and time.time() < deadline:
        ok_now = []
        for k in list(missing):
            if _head_ok(k):
                ok_now.append(k)
        for k in ok_now:
            missing.discard(k)
        if not missing:
            break
        # sleep 渐进
        slp = _WAIT_INTERVALS[min(i, len(_WAIT_INTERVALS)-1)]
        i += 1
        time.sleep(slp)
    if missing:
        raise FileNotFoundError(
            f"[artefacts] offline artefacts not found after waiting {_WAIT_TOTAL_S}s: "
            + ", ".join(missing)
            + f".\n  HINT: ensure the offline step finished and wrote to s3://{BUCKET}/{MODEL_DIR}/"
        )

# ---------- 公开 API：名称保持不变，兼容旧代码 ----------

def load_selected_feats() -> List[str]:
    key = f"{MODEL_DIR}/selected_feats.json"
    _wait_keys([key])
    raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    feats = json.loads(raw.decode())
    if not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
        raise ValueError(f"[artefacts] bad format in {key}")
    return feats

def load_scaler():
    key = f"{MODEL_DIR}/scaler.pkl"
    _wait_keys([key])
    raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    bio = io.BytesIO(raw)
    return joblib.load(bio)

def read_latest() -> Optional[Tuple[str, str, Optional[str]]]:
    """
    返回 (model_key, metrics_key, timestamp_str) 或 None
    latest.txt 形如:
      model_1699999999.pt
      metrics_1699999999.json
    """
    key = f"{MODEL_DIR}/latest.txt"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        txt = obj["Body"].read().decode().strip().splitlines()
        if not txt:
            return None
        model_key   = txt[0].strip()
        metrics_key = (txt[1].strip() if len(txt) >= 2 else "metrics_tmp.json")
        ts = None
        try:
            tsobj = s3.head_object(Bucket=BUCKET, Key=model_key)
            ts = tsobj.get("LastModified") and tsobj["LastModified"].isoformat()
        except Exception:
            pass
        return (model_key, metrics_key, ts)
    except Exception:
        return None

def load_model_by_key(model_key: str):
    import torch
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/{model_key}" if "/" not in model_key else model_key)["Body"].read()
    bio = io.BytesIO(raw)
    mdl = torch.load(bio, map_location="cpu")
    return mdl, raw

def write_latest(model_bytes: bytes, metrics: Dict, model_key: str, metrics_key: str) -> None:
    # 写模型
    save_bytes(f"{MODEL_DIR}/{model_key}", model_bytes, "application/octet-stream")
    # 写指标
    save_bytes(f"{MODEL_DIR}/{metrics_key}", json.dumps(metrics, ensure_ascii=False, indent=2).encode(), "application/json")
    # 指针
    latest = f"{model_key}\n{metrics_key}\n"
    save_bytes(f"{MODEL_DIR}/latest.txt", latest.encode(), "text/plain")
