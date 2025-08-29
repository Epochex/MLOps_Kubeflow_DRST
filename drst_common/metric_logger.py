# /data/mlops/DRST-SoftwarizedNetworks/drst_common/metric_logger.py
# Unified lightweight metric logger for persistence (local write, upload on demand)
import csv
import json
import os
import datetime
import tempfile
import shutil
from typing import List, Dict, Any
from .config import RESULT_DIR
from .minio_helper import save_bytes
import glob

_CSV_PATH  = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
_JSONL_PATH = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.jsonl"

# Default column order (extendable)
_BASE_ORDER: List[str] = [
    "utc", "component", "event",
    "value",
    # training
    "train_rows", "train_time_s", "model_size_mb", "mae", "rmse", "accuracy",
    # inference
    "batch_size", "latency_ms", "model_loading_ms", "cpu_pct", "gpu_mem_pct",
    # drift
    "js_val", "kafka_lag", "update_trigger_delay_s",
    # runtime
    "runtime_ms", "cpu_time_ms",
    # cold start & container
    "cold_start_ms", "rtt_ms", "container_latency_ms",
]

def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)

def _read_all_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))

def _rewrite_csv(path: str, header: List[str], rows: List[Dict[str, Any]]):
    with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(tmp.name, path)

def log_metric(component: str, **kw) -> None:
    """Write metrics to local CSV/JSONL; uploading is not blocked here."""
    _ensure_dir()
    kw["utc"] = kw.get("utc") or datetime.datetime.utcnow().isoformat() + "Z"
    kw["component"] = component

    # Global summary.csv
    rows_all = _read_all_rows(_CSV_PATH)
    header = list(rows_all[0].keys()) if rows_all else _BASE_ORDER.copy()
    for k in kw:
        if k not in header:
            header.append(k)
    rows_all.append({k: kw.get(k, "") for k in header})
    _rewrite_csv(_CSV_PATH, header, rows_all)

    # Component-specific CSV
    comp_path = f"/mnt/pvc/{RESULT_DIR}/{component}_metrics.csv"
    rows_comp = _read_all_rows(comp_path)
    header_comp = list(rows_comp[0].keys()) if rows_comp else header.copy()
    for k in kw:
        if k not in header_comp:
            header_comp.append(k)
    rows_comp.append({k: kw.get(k, "") for k in header_comp})
    _rewrite_csv(comp_path, header_comp, rows_comp)

    # JSONL append
    with open(_JSONL_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(kw, ensure_ascii=False) + "\n")

def sync_all_metrics_to_minio() -> None:
    """Upload all local metric files to MinIO in one batch."""
    files = [
        (_CSV_PATH, "text/csv"),
        (_JSONL_PATH, "application/json")
    ]
    files += [
        (path, "text/csv")
        for path in glob.glob(f"/mnt/pvc/{RESULT_DIR}/*_metrics.csv")
    ]
    for path, mime in files:
        if not os.path.exists(path):
            continue
        key = f"{RESULT_DIR}/{os.path.basename(path)}"
        with open(path, "rb") as f:
            data = f.read()
        save_bytes(key, data, mime)
