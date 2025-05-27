# 全局路径 & Kafka
DATA_DIR        = "datasets"
MODEL_DIR       = "models"
KAFKA_TOPIC     = "latencyTopic"
KAFKA_SERVERS   = ["kafka.default.svc.cluster.local:9092"]

# 训练 / 推理超参
BATCH_SIZE        = 60
CONSUME_TIMEOUT_S = 300
CORRECTION_OFFSET = 38

SEED = 40
TARGET_COL   = "output_rate"
EXCLUDE_COLS = ["Unnamed: 0", "input_rate", "output_rate", "latency"]
