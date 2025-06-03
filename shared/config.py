# shared/config.py

import os

# -------------- MinIO 配置 --------------
ENDPOINT   = os.getenv("MINIO_ENDPOINT",  "45.149.207.13:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY","minio")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY","minio123")
# 默认使用 onvm-demo2 桶
BUCKET     = os.getenv("MINIO_BUCKET",    "onvm-demo2")

# -------------- 数据、模型、结果 保存目录 --------------
# (这些目录是相对于容器工作目录或挂载到 /mnt/pvc 的挂载点)
DATA_DIR    = "datasets"
MODEL_DIR   = "models"
RESULT_DIR  = "results"

# -------------- Kafka 配置 --------------
KAFKA_TOPIC        = "latencyTopic"
KAFKA_SERVERS      = ["kafka.default.svc.cluster.local:9092"]

# Consumer 从 latest offset 开始，跳过旧数据
AUTO_OFFSET_RESET  = "latest"
ENABLE_AUTO_COMMIT = True

# Producer/Monitor 批次大小
BATCH_SIZE         = 10

# Consumer 空闲多久(秒)后自动退出
CONSUME_IDLE_S     = 10

# JS drift 阈值
JS_DEFAULT_THRESH    = 0.2
JS_LOW_THRESHOLD     = 0.4
JS_MEDIUM_THRESHOLD  = 0.6

# 其他常量
TARGET_COL         = "output_rate"
EXCLUDE_COLS       = ["Unnamed: 0", "input_rate", "output_rate", "latency"]
MAPPING_KEY        = f"{RESULT_DIR}/js_accuracy_mapping.json"
