# 全局路径 & Kafka
DATA_DIR        = "datasets"
MODEL_DIR       = "models"
KAFKA_TOPIC     = "latencyTopic"
KAFKA_SERVERS   = ["kafka.default.svc.cluster.local:9092"]

# Kafka
KAFKA_TOPIC    = "latencyTopic"
KAFKA_SERVERS  = ["kafka.default.svc.cluster.local:9092"]

# Stream
BATCH_SIZE     = 60
CONSUME_IDLE_S = 30
CORRECTION_OFFSET = 38

# Data keys in MinIO
DATA_DIR   = "datasets"           # bucket 下 datasets/...
MODEL_DIR  = "models"             # bucket 下 models/...
RESULT_DIR = "results"            # bucket 下 results/...

TARGET_COL = "output_rate"


SEED = 40
TARGET_COL   = "output_rate"
EXCLUDE_COLS = ["Unnamed: 0", "input_rate", "output_rate", "latency"]
