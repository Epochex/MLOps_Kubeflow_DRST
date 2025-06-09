# shared/metric_logger.py
# ------------------------------------------------------------
# 统一落盘的轻量指标记录器（本地写入，按需上传）
# ------------------------------------------------------------
import csv
import json
import os
import datetime
import tempfile
import shutil
from typing import List, Dict, Any

from .config import RESULT_DIR

_CSV_PATH = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
_JSONL_PATH = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.jsonl"

# 默认列顺序（可扩展）
_BASE_ORDER: List[str] = [
    "utc", "component", "event",
    "value",
    # 训练
    "train_rows", "train_time_s", "model_size_mb", "mae", "rmse", "accuracy",
    # 推理
    "batch_size", "latency_ms", "model_loading_ms", "cpu_pct", "gpu_mem_pct",
    # drift
    "js_val", "kafka_lag", "update_trigger_delay_s",
    # 运行时
    "runtime_ms", "cpu_time_ms",
    # 冷启与容器
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
    """
    将指标写入本地 CSV/JSONL，不再实时上传到 MinIO。
    """
    _ensure_dir()
    # 补全字段
    kw["utc"] = kw.get("utc") or datetime.datetime.utcnow().isoformat() + "Z"
    kw["component"] = component

    # 1) 全局 summary.csv
    rows_all = _read_all_rows(_CSV_PATH)
    header = list(rows_all[0].keys()) if rows_all else _BASE_ORDER.copy()
    for k in kw:
        if k not in header:
            header.append(k)
    rows_all.append({k: kw.get(k, "") for k in header})
    _rewrite_csv(_CSV_PATH, header, rows_all)

    # 2) 单 component CSV
    comp_path = f"/mnt/pvc/{RESULT_DIR}/{component}_metrics.csv"
    rows_comp = _read_all_rows(comp_path)
    header_comp = list(rows_comp[0].keys()) if rows_comp else header.copy()
    for k in kw:
        if k not in header_comp:
            header_comp.append(k)
    rows_comp.append({k: kw.get(k, "") for k in header_comp})
    _rewrite_csv(comp_path, header_comp, rows_comp)

    # 3) JSONL 追加
    with open(_JSONL_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(kw, ensure_ascii=False) + "\n")

def sync_all_metrics_to_minio() -> None:
    """
    将所有本地指标文件一次性上传到 MinIO，包括：
      - metrics_summary.csv
      - metrics_summary.jsonl
      - *_metrics.csv（各组件）
    """
    from .minio_helper import save_bytes
    import glob

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
