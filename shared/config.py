# shared/config.py
# ------------------------------------------------------------
# 全局常量 & 运行时环境变量
# ------------------------------------------------------------
import os

# ---------- MinIO ----------
ENDPOINT   = os.getenv("MINIO_ENDPOINT",  "45.149.207.13:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET     = os.getenv("MINIO_BUCKET",     "onvm-demo2")

# ---------- 项目目录 ----------
DATA_DIR   = "datasets"
MODEL_DIR  = "models"
RESULT_DIR = "results"

# ---------- Kafka ----------
KAFKA_TOPIC        = "latencyTopic"
KAFKA_SERVERS      = ["kafka.default.svc.cluster.local:9092"]

AUTO_OFFSET_RESET  = "latest"
ENABLE_AUTO_COMMIT = True
BATCH_SIZE         = 10
CONSUME_IDLE_S     = 10

# ---------- Drift 监控 ----------
JS_DEFAULT_THRESH   = 0.20
JS_LOW_THRESHOLD    = 0.40
JS_MEDIUM_THRESHOLD = 0.60          # 这里只是参考值，真正分段逻辑在 dynamic_retrain.py

# ---------- 预测目标 ----------
TARGET_COL   = "input_rate"         # ← 改成 input_rate
EXCLUDE_COLS = ["Unnamed: 0", "output_rate", "latency"]  # 去掉 input_rate

# ---------- 其他 ----------
MAPPING_KEY  = f"{RESULT_DIR}/js_accuracy_mapping.json"
