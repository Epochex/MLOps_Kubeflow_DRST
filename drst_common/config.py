import os
from typing import Optional, List, Dict, Tuple

def _to_bool(s: str | None, default: bool = True) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "on")

# MinIO
MINIO_ACCESS_MODE = os.getenv("MINIO_ACCESS_MODE", "ingress").lower()
_INGRESS_DEFAULT = "minio.45.149.207.13.nip.io:30080"
_CLUSTER_DEFAULT = "minio-service.kubeflow.svc.cluster.local:9000"

ENDPOINT = os.getenv("MINIO_API_ENDPOINT") or os.getenv(
    "MINIO_ENDPOINT",
    _INGRESS_DEFAULT if MINIO_ACCESS_MODE == "ingress" else _CLUSTER_DEFAULT
)
MINIO_SCHEME = os.getenv("MINIO_SCHEME", "http").lower()
MINIO_VERIFY_SSL = _to_bool(os.getenv("MINIO_VERIFY_SSL", "true"), default=True)

ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET     = os.getenv("MINIO_BUCKET",     "onvm-demo2")
MINIO_CONSOLE_URL = os.getenv("MINIO_CONSOLE_URL", "http://minio.45.149.207.13.nip.io:30080/")

# Kafka
KAFKA_TOPIC        = os.getenv("KAFKA_TOPIC", "latencyTopic")
KAFKA_SERVERS      = os.getenv("KAFKA_SERVERS", "kafka.default.svc.cluster.local:9092").split(",")
AUTO_OFFSET_RESET  = os.getenv("AUTO_OFFSET_RESET", "latest")
ENABLE_AUTO_COMMIT = _to_bool(os.getenv("ENABLE_AUTO_COMMIT", "true"), default=True)
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "5"))
CONSUME_IDLE_S     = int(os.getenv("CONSUME_IDLE_S", "5"))

# Project paths
DATA_DIR   = os.getenv("DATA_DIR",   "datasets")
MODEL_DIR  = os.getenv("MODEL_DIR",  "models")
RESULT_DIR = os.getenv("RESULT_DIR", "results")

# Feature schema
FEATURE_SRC_KEY = os.getenv("FEATURE_SRC_KEY", f"{DATA_DIR}/combined.csv")

# Metrics threshold
ACC_THR = float(os.getenv("ACC_THR", "0.25"))

# Columns
TARGET_COL   = os.getenv("TARGET_COL", "output_rate")
EXCLUDE_COLS = ["Unnamed: 0", "input_rate", "latency"]

# Drift
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "300"))
EVAL_STRIDE  = int(os.getenv("EVAL_STRIDE",  "50"))

# Producer timing
PRODUCE_INTERVAL_MS = int(os.getenv("PRODUCE_INTERVAL_MS", "100"))

# Offline train/eval policy
OFFLINE_TRAIN_KEY = os.getenv("OFFLINE_TRAIN_KEY", f"{DATA_DIR}/combined.csv")
OFFLINE_EVAL_SOURCE_KEY = os.getenv("OFFLINE_EVAL_SOURCE_KEY", OFFLINE_TRAIN_KEY)
OFFLINE_EVAL_ROWS  = int(os.getenv("OFFLINE_EVAL_ROWS", "500"))
OFFLINE_EVAL_TAKE  = os.getenv("OFFLINE_EVAL_TAKE", "random").lower()
OFFLINE_EVAL_SEED  = int(os.getenv("OFFLINE_EVAL_SEED", "42"))

# Producer stages (rich form): {"key":..., "rows":..., "take": head|tail|random, "seed": optional}
def _parse_legacy_env_to_rich() -> List[Dict]:
    raw = os.getenv("PRODUCER_STAGES", "").strip()
    out: List[Dict] = []
    if not raw:
        return out
    for chunk in [c.strip() for c in raw.split(",") if c.strip()]:
        if ":" not in chunk:
            continue
        k, n = chunk.split(":", 1)
        try:
            out.append({"key": k.strip(), "rows": int(n.strip()), "take": "head"})
        except ValueError:
            pass
    return out

PRODUCER_STAGES: List[Dict] = [
    {"key": f"{DATA_DIR}/combined.csv",                               "rows": 500,  "take": "tail"},
    {"key": f"{DATA_DIR}/random_rates.csv",                           "rows": 1000, "take": "head"},
    {"key": f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv","rows": 1000, "take": "head"},
]

try:
    import json as _json
    _ps_json = os.getenv("PRODUCER_STAGES_JSON", "").strip()
    if _ps_json:
        maybe = _json.loads(_ps_json)
        if isinstance(maybe, list) and maybe:
            PRODUCER_STAGES = maybe
    elif os.getenv("PRODUCER_STAGES"):
        legacy = _parse_legacy_env_to_rich()
        if legacy:
            PRODUCER_STAGES = legacy
except Exception:
    pass

# Retrain thresholds
RETRAIN_JS_NO_RETRAIN = float(os.getenv("RETRAIN_JS_NO_RETRAIN", "0.40"))
RETRAIN_JS_GRID_A     = float(os.getenv("RETRAIN_JS_GRID_A",     "0.60"))
RETRAIN_JS_GRID_B     = float(os.getenv("RETRAIN_JS_GRID_B",     "0.75"))

# ABC grids
def abc_grids(current_hidden: Optional[List[int]] = None) -> Dict[str, Dict[str, list]]:
    base_hidden = tuple(current_hidden or [128, 64, 32])
    return {
        "A": {
            "learning_rate": [1e-3, 5e-4, 1e-4],
            "batch_size": [16, 32],
            "hidden_layers": [base_hidden],
            "activation": ["relu"],
            "loss": ["mse"],
            "wd": [0.0],
            "topk": 1,
        },
        "B": {
            "learning_rate": [1e-3, 5e-4],
            "batch_size": [16, 32],
            "hidden_layers": [(64, 32), (128, 64, 32)],
            "activation": ["relu", "tanh"],
            "loss": ["mse"],
            "wd": [0.0],
            "topk": 2,
        },
        "C": {
            "learning_rate": [1e-2, 1e-3],
            "batch_size": [16, 32, 64],
            "hidden_layers": [(256, 128, 64), (128, 128, 64, 32)],
            "activation": ["relu", "gelu"],
            "loss": ["huber", "mse"],
            "wd": [0.0, 1e-3],
            "topk": 3,
        },
    }
