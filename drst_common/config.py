#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, List, Optional, Union
import os

# ===== MinIO / S3 =====
MINIO_ACCESS_MODE = "cluster"
MINIO_SCHEME      = "http"
MINIO_ENDPOINT    = "minio-service.kubeflow.svc.cluster.local:9000"
BUCKET            = "onvm-demo2"

MODEL_DIR  = "models"
RESULT_DIR = "results"
DATA_DIR   = "datasets"

# ===== Kafka =====
KAFKA_SERVERS = "kafka.default.svc.cluster.local:9092"
KAFKA_TOPIC   = "latencyTopic"

# ===== 特征 / 训练（离线）=====
FEATURE_SRC_KEY = f"{DATA_DIR}/combined.csv"
EXCLUDE_COLS    = ["Unnamed: 0", "input_rate", "latency"]
TARGET_COL      = "output_rate"
OFFLINE_TOPK    = 10
ACC_THR         = 0.25   # 累计命中阈值（infer 打 cum@ 时用）

# ===== 批大小（逐条推理）=====
BATCH_SIZE = 1

# ===== Consumer（在线推理）=====
CONSUME_IDLE_S     = 30
RELOAD_INTERVAL_S  = 30
INFER_STDOUT_EVERY = 1
GAIN_THR_PP        = 0.01
RETRAIN_TOPIC      = KAFKA_TOPIC + "_infer_count"

# infer 打印：本批命中阈值（可改）
INFER_HIT_THR = 0.15

# ===== Producer（速率与分区策略）=====
# 速率优先级：env(PRODUCER_TPS) > env(INTERVAL_MS) > PRODUCER_TPS/PRODUCE_INTERVAL_MS
PRODUCER_TPS        = 10.0          # 每秒 5 条（可被 env 覆盖）
PRODUCE_INTERVAL_MS = 100          # 兼容老参数，若无 TPS 则用它
PRODUCER_JITTER_MS  = 0
PRODUCER_PARTITION_MODE = "rr"     # "auto" | "rr" | "hash"

# 分段条数
PRODUCER_BRIDGE_N = 500
PRODUCER_RAND_N   = 1000
PRODUCER_STIM_N   = 1000

PRODUCER_STAGES: List[dict] = [
    {"key": f"{DATA_DIR}/combined.csv",     "take": "tail", "rows": PRODUCER_BRIDGE_N},
    {"key": f"{DATA_DIR}/random_rates.csv", "take": "head", "rows": PRODUCER_RAND_N},
    {"key": f"{DATA_DIR}/resource_stimulus_global_A-B-C_modified.csv", "take": "head", "rows": PRODUCER_STIM_N},
]

# ===== 漂移监控窗口 / 阈值 =====
DRIFT_WINDOW = 300
EVAL_STRIDE  = 50
HIST_BINS    = 64
JS_THR_A     = 0.7
JS_THR_B     = 0.3
JS_THR_C     = 0.8

# monitor 把“最新 JS”广播到这个对象键；infer 实时读取
JS_CURRENT_KEY = f"{RESULT_DIR}/js_current.json"

# ===== Monitor 其它 =====
WAIT_FEATURES_SECS      = 120
MONITOR_IDLE_TIMEOUT_S  = 60

# ---- 统一全局墙时长入口（替代过去“导入即绑定”的常量）----
def _parse_int_like(x, default: int) -> int:
    try:
        return int(float(str(x)))
    except Exception:
        return default

# 默认值（未设环境变量时的兜底）。你也可以通过环境变量 DEFAULT_MAX_WALL_SECS 覆盖。
DEFAULT_MAX_WALL_SECS: int = _parse_int_like(os.getenv("DEFAULT_MAX_WALL_SECS", None), -1)

# 快照：模块导入时根据环境变量计算一次（兼容旧代码直接引用 MAX_WALL_SECS）
# 兼容旧名：RETRAIN_MAX_WALL_SECS
MAX_WALL_SECS: int = _parse_int_like(
    os.getenv("MAX_WALL_SECS", os.getenv("RETRAIN_MAX_WALL_SECS", None)),
    DEFAULT_MAX_WALL_SECS,
)

# 进程内临时覆盖（测试/热切换）；None 取消覆盖
_MAX_WALL_OVERRIDE: Optional[int] = None

def set_max_wall_secs(value: Optional[Union[int, float, str]]) -> None:
    """
    在当前进程内覆盖墙时长；传 None 取消覆盖。
    优先级：set_* 覆盖 > 环境变量(MAX_WALL_SECS/RETRAIN_MAX_WALL_SECS) > 模块导入快照
    """
    global _MAX_WALL_OVERRIDE
    if value is None:
        _MAX_WALL_OVERRIDE = None
    else:
        _MAX_WALL_OVERRIDE = _parse_int_like(value, DEFAULT_MAX_WALL_SECS)

def get_max_wall_secs() -> int:
    """
    统一入口：任何组件需要墙时长都应调用本函数，而不是直接 import 常量。
    返回 <=0 表示不设上限。
    """
    if _MAX_WALL_OVERRIDE is not None:
        return _MAX_WALL_OVERRIDE
    env = os.getenv("MAX_WALL_SECS") or os.getenv("RETRAIN_MAX_WALL_SECS")
    if env is not None:
        try:
            return int(float(env))
        except Exception:
            pass
    return MAX_WALL_SECS
# ---- 统一全局墙时长入口（END）----

# 当触发重训时，monitor 是否给 infer 下发“暂停”旗标（默认 False：不暂停，保证预测连续性）
MONITOR_SIGNAL_INFER_PAUSE = False
# infer 是否响应暂停旗标（默认 True）
INFER_RESPECT_PAUSE_FLAG   = True
# 暂停旗标对象键
PAUSE_INFER_KEY = f"{RESULT_DIR}/pause_infer.flag"
# 老名字，避免他处引用报错；锁定逻辑现在始终开启
MONITOR_WAIT_RETRAIN = False

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

# ===== Retrain =====
RETRAIN_WARM_EPOCHS    = 3
RETRAIN_EPOCHS_FULL    = 8
RETRAIN_EARLY_PATIENCE = 2
RETRAIN_MODE           = "auto"   # "scratch" | "finetune" | "auto"
RETRAIN_FREEZE_N       = 0
RETRAIN_VAL_FRAC       = 0.2

# ===== 动态重训网格 =====
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
            "learning_rate": [5e-3, 1e-2],
            "batch_size":    [16, 32],
            "hidden_layers": base_small,
            "activation":    ["relu", "gelu"],
            "loss":          ["smoothl1"],
            "wd":            [0.0, 1e-4],
            "topk":          2,
        },
        "B": {
            "learning_rate": [3e-3, 1e-2],
            "batch_size":    [16, 32],
            "hidden_layers": base_medium,
            "activation":    ["relu", "gelu"],
            "loss":          ["smoothl1", "mse"],
            "wd":            [0.0, 1e-4],
            "topk":          3,
        },
        "C": {
            "learning_rate": [1e-3, 3e-3, 1e-2],
            "batch_size":    [16, 32],
            "hidden_layers": base_large,
            "activation":    ["relu", "gelu"],
            "loss":          ["smoothl1", "mse"],
            "wd":            [0.0, 1e-4],
            "topk":          3,
        },
    }
