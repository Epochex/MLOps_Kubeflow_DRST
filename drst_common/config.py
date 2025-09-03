# /data/mlops/DRST-SoftwarizedNetworks/drst_common/config.py
import os
from typing import Optional, List, Dict

def _to_bool(s: str | None, default: bool = True) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "on")

# ---------------- MinIO ----------------
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

# ---------------- Kafka ----------------
KAFKA_TOPIC        = os.getenv("KAFKA_TOPIC", "latencyTopic")
KAFKA_SERVERS      = os.getenv("KAFKA_SERVERS", "kafka.default.svc.cluster.local:9092").split(",")
AUTO_OFFSET_RESET  = os.getenv("AUTO_OFFSET_RESET", "latest")
ENABLE_AUTO_COMMIT = _to_bool(os.getenv("ENABLE_AUTO_COMMIT", "true"), default=True)
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "5"))
CONSUME_IDLE_S     = int(os.getenv("CONSUME_IDLE_S", "5"))

# ---------------- Project paths ----------------
DATA_DIR   = os.getenv("DATA_DIR",   "datasets")
MODEL_DIR  = os.getenv("MODEL_DIR",  "models")
RESULT_DIR = os.getenv("RESULT_DIR", "results")

# ---------------- Targets / Columns ----------------
TARGET_COL   = os.getenv("TARGET_COL", "output_rate")
EXCLUDE_COLS = ["Unnamed: 0", "input_rate", "latency"]

# ---------------- Drift controls ----------------
# Fixed sliding window = 300, recompute JS every 50 new records
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "300"))
EVAL_STRIDE  = int(os.getenv("EVAL_STRIDE",  "50"))

# === Retrain triggers & thresholds (centralized; can override with env vars, no JSON involved) ===
RETRAIN_JS_NO_RETRAIN = float(os.getenv("RETRAIN_JS_NO_RETRAIN", "0.40"))  # < 0.40 → no retrain
RETRAIN_JS_GRID_A     = float(os.getenv("RETRAIN_JS_GRID_A",     "0.60"))  # [0.40,0.60) → A
RETRAIN_JS_GRID_B     = float(os.getenv("RETRAIN_JS_GRID_B",     "0.75"))  # [0.60,0.75) → B; otherwise C

# ---------------- ABC grid centralized definition ----------------
def abc_grids(current_hidden: Optional[List[int]] = None) -> Dict[str, Dict[str, list]]:
    """
    Return the parameter search spaces for A/B/C grids.
    - Does not read/require any JSON; to change grids, edit here for global effect.
    - A: tune only lr/batch, reuse current model structure; if unavailable, fallback to (128,64,32).
    """
    base_hidden = tuple(current_hidden or [128, 64, 32])
    return {
        "A": {
            "learning_rate": [1e-3, 5e-4, 1e-4],
            "batch_size": [16, 32],
            "hidden_layers": [base_hidden],
            "activation": ["relu"],
            "loss": ["mse"],
            "wd": [0.0],
            "topk": 1,  # A is already lightweight, directly train full model after warm scoring
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
