# shared/metric_logger.py
import csv, json, os, datetime
from typing import List

from .config import RESULT_DIR
from .minio_helper import save_bytes

_CSV_PATH   = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
_JSONL_PATH = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.jsonl"

# 默认字段顺序（会自动扩展）
_BASE_ORDER: List[str] = [
    "utc", "component", "event",
    # ---- 通用 ----
    "value",
    # ---- 训练 ----
    "train_rows", "train_time_s", "model_size_mb",
    "mae", "rmse", "accuracy",
    # ---- 推理 ----
    "batch_size", "latency_ms", "model_loading_ms",
    "cpu_pct", "gpu_mem_pct",
    # ---- drift ----
    "js_val", "kafka_lag", "update_trigger_delay_s",
]

def _ensure_dir():
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)

def log_metric(component: str, **kw):
    """
    任何字段都能写进来；若字段原先不存在，会自动加入 CSV 表头。
    用法:  log_metric(component="monitor", event="drift_calc", js_val=0.23, cpu_pct=12.3)
    """
    _ensure_dir()

    kw["utc"] = kw.get("utc") or datetime.datetime.utcnow().isoformat() + "Z"
    kw["component"] = component

    # -------------------------------------------------
    # 1️⃣ 动态确定字段顺序
    if os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, newline="") as fp:
            reader = csv.reader(fp)
            header: List[str] = next(reader)
    else:
        header = _BASE_ORDER.copy()

    for k in kw:
        if k not in header:
            header.append(k)

    # -------------------------------------------------
    # 2️⃣  写 CSV
    is_new = not os.path.exists(_CSV_PATH)
    with open(_CSV_PATH, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({k: kw.get(k, "") for k in header})

    # -------------------------------------------------
    # 3️⃣  写 JSONL
    with open(_JSONL_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(kw, ensure_ascii=False) + "\n")

    # -------------------------------------------------
    # 4️⃣  上传 MinIO
    for path, mime in [
        (_CSV_PATH,  "text/csv"),
        (_JSONL_PATH,"application/json"),
    ]:
        save_bytes(f"{RESULT_DIR}/{os.path.basename(path)}",
                   open(path, "rb").read(), mime)
