# shared/config.py

import os

# ---------- MinIO 连接信息 ----------
ENDPOINT   = os.getenv("MINIO_ENDPOINT",  "45.149.207.13:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET     = os.getenv("MINIO_BUCKET",     "onvm-demo2")

# ---------- 项目目录配置 ----------
# 本地代码里所有数据 / 模型 / 结果都挂在这几个路径下面
DATA_DIR   = "datasets"
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
TARGET_COL   = "input_rate"
EXCLUDE_COLS = ["Unnamed: 0", "output_rate", "latency"]

# ---------- Drift 监控 & Adaptation 阈值统一管理 ----------
# 监控模块：只要 JS > 该阈值 就调用 retrain
JS_TRIGGER_THRESH = 0.2  

# Adaptation 模块：不同 js_val 区间对应不同网格
JS_SEV1_THRESH    = 0.4   # ≤0.4 → 轻量网格 A
JS_SEV2_THRESH    = 0.6   # ≤0.6 → 中等网格 B
# >0.6 → 全网格 C

# ---------- 其他（用到 RESULT_DIR 的地方） ----------
MAPPING_KEY  = f"{RESULT_DIR}/js_accuracy_mapping.json"
