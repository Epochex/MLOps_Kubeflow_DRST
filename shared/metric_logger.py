# shared/metric_logger.py
# ------------------------------------------------------------
# 统一落盘 + MinIO 同步的轻量指标记录器
# ❶ 新增：每个 component 也各写一份 <component>_metrics.csv
# ------------------------------------------------------------
import csv, json, os, datetime, tempfile, shutil
from typing import List, Dict, Any

from .config       import RESULT_DIR
from .minio_helper import save_bytes

_CSV_PATH   = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.csv"
_JSONL_PATH = f"/mnt/pvc/{RESULT_DIR}/metrics_summary.jsonl"

# —— 默认列顺序（后续会动态扩展） ——————————————
_BASE_ORDER: List[str] = [
    "utc", "component", "event",
    # ---- 通用 ----------
    "value",
    # ---- 训练 ----------
    "train_rows", "train_time_s", "model_size_mb",
    "mae", "rmse", "accuracy",
    # ---- 推理 ----------
    "batch_size", "latency_ms", "model_loading_ms",
    "cpu_pct", "gpu_mem_pct",
    # ---- drift ----------
    "js_val", "kafka_lag", "update_trigger_delay_s",
    # ---- Runtime profiling ----------
    "runtime_ms", "cpu_time_ms",
    # ---- 冷启 & 容器 ----------
    "cold_start_ms", "rtt_ms", "container_latency_ms",
]

# ------------------------------------------------------------
def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)

def _read_all_rows(path: str) -> list[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))

def _rewrite_csv(path: str, header: List[str], rows: List[Dict[str, Any]]):
    """
    原子方式重写 CSV（确保表头始终最新 & “列 ↔ 数据” 对齐）
    """
    with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(tmp.name, path)

def _sync_to_minio(rel_path: str, mime: str):
    """
    把本地文件同步到 MinIO（Key 与本地相同层级）
    """
    save_bytes(f"{RESULT_DIR}/{os.path.basename(rel_path)}",
               open(rel_path, "rb").read(), mime)


def log_metric(component: str, **kw) -> None:
    """
    通用埋点：任何字段都能写；出现新字段会自动更新所有文件表头
    """
    _ensure_dir()

    # —— 必备字段 ——  
    kw["utc"]       = kw.get("utc") or datetime.datetime.utcnow().isoformat() + "Z"
    kw["component"] = component

    # —— ① 全局 summary.csv —— ------------------------------------------
    rows_all  = _read_all_rows(_CSV_PATH)
    header: List[str] = list(rows_all[0].keys()) if rows_all else _BASE_ORDER.copy()

    for k in kw:            # 扩展新列
        if k not in header:
            header.append(k)

    rows_all.append({k: kw.get(k, "") for k in header})
    _rewrite_csv(_CSV_PATH, header, rows_all)
    _sync_to_minio(_CSV_PATH, "text/csv")

    # —— ② 单 component CSV —— -----------------------------------------
    comp_path = f"/mnt/pvc/{RESULT_DIR}/{component}_metrics.csv"
    rows_comp = _read_all_rows(comp_path)
    header_comp: List[str] = list(rows_comp[0].keys()) if rows_comp else header.copy()

    for k in kw:
        if k not in header_comp:
            header_comp.append(k)

    rows_comp.append({k: kw.get(k, "") for k in header_comp})
    _rewrite_csv(comp_path, header_comp, rows_comp)
    _sync_to_minio(comp_path, "text/csv")

    # —— ③ 追加 JSONL —— -----------------------------------------------
    with open(_JSONL_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(kw, ensure_ascii=False) + "\n")
    _sync_to_minio(_JSONL_PATH, "application/json")
