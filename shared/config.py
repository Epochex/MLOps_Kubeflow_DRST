# shared/config.py

import os

# ---------- MinIO 连接信息 ----------
ENDPOINT   = os.getenv("MINIO_ENDPOINT",  "45.149.207.13:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET     = os.getenv("MINIO_BUCKET",     "onvm-demo2")

# ---------- 项目目录配置 ----------
DATA_DIR   = "datasets_old"
MODEL_DIR  = "models"
RESULT_DIR = "results"

# ---------- Kafka 配置 ----------
KAFKA_TOPIC        = "latencyTopic"
KAFKA_SERVERS      = ["kafka.default.svc.cluster.local:9092"]
AUTO_OFFSET_RESET  = "latest"
ENABLE_AUTO_COMMIT = True
BATCH_SIZE         = 5
CONSUME_IDLE_S     = 5

# ---------- 预测目标列名 ----------
TARGET_COL   = "output_rate"
EXCLUDE_COLS = ["Unnamed: 0", "input_rate", "latency"]

# ---------- Drift 监控 & Adaptation 阈值（可由环境变量覆盖） ----------
JS_TRIGGER_THRESH = float(os.getenv("JS_TRIGGER_THRESH", "0.20"))
JS_SEV1_THRESH    = float(os.getenv("JS_SEV1_THRESH",    "0.50"))
JS_SEV2_THRESH    = float(os.getenv("JS_SEV2_THRESH",    "0.80"))


# ---------- 其他（用到 RESULT_DIR 的地方） ----------
MAPPING_KEY  = f"{RESULT_DIR}/js_accuracy_mapping.json"
