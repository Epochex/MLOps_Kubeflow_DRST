#!/usr/bin/env python3
# drst_common/config.py
from __future__ import annotations
from typing import Dict, List

# ===== MinIO / S3 =====
MINIO_ACCESS_MODE = "cluster"
MINIO_SCHEME      = "http"
MINIO_ENDPOINT    = "minio-service.kubeflow.svc.cluster.local:9000"
BUCKET            = "onvm-demo2"

# 目录前缀
MODEL_DIR  = "models"
RESULT_DIR = "results"
DATA_DIR   = "datasets"

# ===== Kafka =====
KAFKA_SERVERS = "kafka.default.svc.cluster.local:9092"
KAFKA_TOPIC   = "latencyTopic"

# ===== 特征/训练（离线）=====
FEATURE_SRC_KEY = f"{DATA_DIR}/combined.csv"   # 用于派生完整 60 维 FEATURE_COLS 的“真理源”
EXCLUDE_COLS    = ["Unnamed: 0", "input_rate", "latency"]
TARGET_COL      = "output_rate"
OFFLINE_TOPK    = 10
ACC_THR         = 0.25

# ===== 批大小（逐条推理）=====
BATCH_SIZE = 1  # producer / consumer 同步读取

# ===== Consumer 行为（在线推理）=====
CONSUME_IDLE_S     = 300     # 无数据超时退出
RELOAD_INTERVAL_S  = 30      # 热更新探测周期（秒）
INFER_STDOUT_EVERY = 1       # 每处理几批打印一行
GAIN_THR_PP        = 0.01    # 新模型相对 baseline 的提升阈值（百分点）
RETRAIN_TOPIC      = KAFKA_TOPIC + "_infer_count"

# ===== Producer 行为 =====
PRODUCE_INTERVAL_MS     = 100
PRODUCER_PARTITION_MODE = "rr"  # "auto" | "rr" | "hash"
MANIFEST_KEY            = f"{DATA_DIR}/online/manifest.json"
PRODUCER_STAGES: List[dict] = [
    {"key": f"{DATA_DIR}/combined.csv",     "rows": 1000, "take": "head"},
    {"key": f"{DATA_DIR}/random_rates.csv", "rows": 1000, "take": "head"},
]

# ===== 漂移监控窗口 =====
DRIFT_WINDOW = 300
EVAL_STRIDE  = 50

# ===== Offline 训练控制 =====
TRAIN_TRIGGER     = 1
OFFLINE_TRAIN_KEY = f"{DATA_DIR}/combined.csv"

FAST_MAX_EPOCH  = 10
FAST_PATIENCE   = 4
FAST_LR         = 1e-3
FAST_BS         = 64

FULL_MAX_EPOCH  = 100
FULL_PATIENCE   = 10
FULL_LR         = 1e-3
FULL_BS         = 16

# ===== Retrain（明文，不读环境变量）=====
RETRAIN_WARM_EPOCHS    = 3
RETRAIN_EPOCHS_FULL    = 8
RETRAIN_EARLY_PATIENCE = 2
RETRAIN_MODE           = "auto"   # "scratch" | "finetune" | "auto"
RETRAIN_FREEZE_N       = 0        # finetune 时可冻结前 N 个 Linear
RETRAIN_VAL_FRAC       = 0.2

# ===== 动态重训网格（保持原样）=====
def abc_grids(current_hidden: List[int] | None = None) -> Dict[str, Dict]:
    def uniq_layers(cands: List[List[int]]) -> List[List[int]]:
        seen = set(); out = []
        for h in cands:
            k = tuple(h)
            if k in seen: continue
            seen.add(k); out.append(h)
        return out

    base_small  = [[32, 16], [64, 32]]
    base_medium = [[64, 32], [96, 48], [128, 64]]
    base_large  = [[128, 64], [160, 80], [192, 96]]

    if current_hidden and isinstance(current_hidden, list) and len(current_hidden) > 0:
        base_small  = uniq_layers([current_hidden] + base_small)
        base_medium = uniq_layers([current_hidden] + base_medium)
        base_large  = uniq_layers([current_hidden] + base_large)

    return {
        "A": {
            "learning_rate": [5e-4, 1e-3],
            "batch_size":    [32, 64],
            "hidden_layers": base_small,
            "activation":    ["relu"],
            "loss":          ["smoothl1"],
            "wd":            [0.0],
            "topk":          1,
        },
        "B": {
            "learning_rate": [5e-4, 1e-3],
            "batch_size":    [32, 64, 128],
            "hidden_layers": base_medium,
            "activation":    ["relu", "gelu"],
            "loss":          ["smoothl1", "mse"],
            "wd":            [0.0, 1e-5],
            "topk":          2,
        },
        "C": {
            "learning_rate": [3e-4, 5e-4, 1e-3],
            "batch_size":    [64, 128],
            "hidden_layers": base_large,
            "activation":    ["relu", "gelu"],
            "loss":          ["smoothl1", "mse"],
            "wd":            [0.0, 1e-5],
            "topk":          2,
        },
    }
