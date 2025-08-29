#!/usr/bin/env python3
# drst_common/runtime.py
# Runtime utilities: ready flag + KFP metadata
from __future__ import annotations
import os
from .minio_helper import save_bytes
from .config import RESULT_DIR

def touch_ready(component: str, name: str = "ready") -> str:
    """local ready flag + upload to PVC/MinIO for external check"""
    tmp_dir = f"/tmp/{component}"
    os.makedirs(tmp_dir, exist_ok=True)
    local = os.path.join(tmp_dir, f"{component}_{name}.flag")
    open(local, "w").close()
    save_bytes(f"{RESULT_DIR}/{component}_{name}.flag", b"", "text/plain")
    return local

def write_kfp_metadata(payload: bytes | None = None) -> None:
    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "wb").write(payload or b"{}")
