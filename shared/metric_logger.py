# shared/metric_logger.py
# ------------------------------------------------------------
# 统一落盘 + MinIO 同步的轻量指标记录器
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

def _read_all_rows() -> list[Dict[str, Any]]:
    if not os.path.exists(_CSV_PATH):
        return []
    with open(_CSV_PATH, newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)

def log_metric(component: str, **kw) -> None:
    """
    通用埋点：任何字段都能写；如出现新字段，
    会自动补齐旧行并更新表头，保证 CSV 永远不会“列名 ↔ 数据” 错位。
    """
    _ensure_dir()

    # —— 必备字段 ——  
    kw["utc"]       = kw.get("utc") or datetime.datetime.utcnow().isoformat() + "Z"
    kw["component"] = component

    # —— ① 读旧文件 ——  
    rows  = _read_all_rows()
    header: List[str] = list(rows[0].keys()) if rows else _BASE_ORDER.copy()   # ★ 关键：转成 list

    # —— ② 扩展表头（发现新字段） ——  
    for k in kw:
        if k not in header:
            header.append(k)

    # —— ③ 把本次记录加入 rows ——  
    rows.append({k: kw.get(k, "") for k in header})

    # —— ④ *整文件重写* CSV（临时 -> 原子替换） ——  
    with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(tmp.name, _CSV_PATH)

    # —— ⑤ 追加 JSONL ——  
    with open(_JSONL_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(kw, ensure_ascii=False) + "\n")

    # —— ⑥ 上传 MinIO ——  
    for path, mime in [
        (_CSV_PATH,  "text/csv"),
        (_JSONL_PATH,"application/json"),
    ]:
        save_bytes(f"{RESULT_DIR}/{os.path.basename(path)}",
                   open(path, "rb").read(), mime)
