# DRST-SoftwarizedNetworks/drst_common/config.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional, Union
import os

"""
本文件是 DRST 的**唯一显式配置入口**：
- S3/MinIO、Kafka、离线/在线/重训/监控的所有默认参数，都在这里定义
- 其余代码应通过导入本文件读取配置，而非依赖外部 export
- 仅墙时长允许在运行期用环境变量或 set_* 覆盖（见文末的 MAX_WALL_SECS 接口）
"""

# ===== MinIO / S3（显式配置）=====
MINIO_ACCESS_MODE = "cluster"  # cluster / local（仅用于你们内部语义）
MINIO_SCHEME      = "http"     # http / https
MINIO_ENDPOINT    = "minio-service.kubeflow.svc.cluster.local:9000"  # 不含 scheme
BUCKET            = "onvm-demo2"

# 目录（S3 前缀）
MODEL_DIR  = "models"
RESULT_DIR = "results"
DATA_DIR   = "datasets"

# ===== Kafka（显式配置）=====
KAFKA_SERVERS = "kafka.default.svc.cluster.local:9092"
KAFKA_TOPIC   = "latencyTopic"

# ===== 特征 / 训练（离线）=====
FEATURE_SRC_KEY = f"{DATA_DIR}/combined.csv"
EXCLUDE_COLS    = ["Unnamed: 0", "input_rate", "latency"]
TARGET_COL      = "output_rate"
OFFLINE_TOPK    = 10
ACC_THR         = 0.25  # 累计命中阈值（infer 里累积准确率使用）

# ===== 在线推理（Consumer）=====
BATCH_SIZE          = 1
CONSUME_IDLE_S      = 30
RELOAD_INTERVAL_S   = 30
INFER_STDOUT_EVERY  = 1
GAIN_THR_PP         = 0.01
RETRAIN_TOPIC       = KAFKA_TOPIC + "_infer_count"
INFER_HIT_THR       = 0.15  # 批内“命中”阈值（打印/指标用）

# ===== Producer（速率与分段）=====
# 说明：runtime 若未传入 pipeline 参数，则按本文件配置执行；无需依赖环境变量。
PRODUCER_TPS        = 10.0   # 若 >0，则大约每秒条数 ≈ TPS
PRODUCE_INTERVAL_MS = 100    # 若 TPS<=0，则使用固定间隔（毫秒）
PRODUCER_JITTER_MS  = 0
PRODUCER_PARTITION_MODE = "rr"  # "auto" | "rr" | "hash"

# ---- 各阶段默认条数（可按需调整）----
# stage0：regular traffic（combined，给 offline；这里也作为在线的第 0 段推流）
PRODUCER_BRIDGE_N = 500
# stage1：random rates
PRODUCER_RAND_N   = 1000
# stage2：resource stimulus（CPU 资源争用）
PRODUCER_STIM_N   = 1000
# stage3：intervention（流量+资源均不稳定）
PRODUCER_INTERV_N = 1000

# ---- 在线按顺序发送的阶段清单 ----
# 说明：
# - name：阶段名（用于日志/消息里的 "stage" 字段，以及环境变量 PRODUCER_STAGES 白名单）
# - key ：MinIO 对象键（**指向 perf 预处理/合并后的产物**）
# - take：从 CSV 取哪一端（"head" | "tail"）
# - rows：取多少行（<=0 表示取尽）
PRODUCER_STAGES: List[dict] = [
    # stage0：combined 作为 warm-up/基线（你之前的“bridge”）
    {"name": "stage0", "key": f"{DATA_DIR}/combined.csv", "take": "tail", "rows": PRODUCER_BRIDGE_N},

    # stage1：random（来自 perf 预处理合并后的全量；Producer 内按 rows 截取）
    {"name": "stage1", "key": f"{DATA_DIR}/perf/stage1_random_rates.csv", "take": "head", "rows": PRODUCER_RAND_N},

    # stage2：resource_stimulus（A-B-C_modified 别名文件）
    {"name": "stage2", "key": f"{DATA_DIR}/perf/stage2_resource_stimulus_global_A-B-C_modified.csv", "take": "head", "rows": PRODUCER_STIM_N},

    # stage3：intervention
    {"name": "stage3", "key": f"{DATA_DIR}/perf/stage3_intervention_global.csv", "take": "head", "rows": PRODUCER_INTERV_N},
]

# ===== 漂移监控窗口 / 阈值 =====
DRIFT_WINDOW = 300
EVAL_STRIDE  = 50
HIST_BINS    = 64
JS_THR_A     = 0.2
JS_THR_B     = 0.5
JS_THR_C     = 0.8

# monitor 广播“最新 JS”的对象键；infer 侧会按需读取
JS_CURRENT_KEY = f"{RESULT_DIR}/js_current.json"

# ===== Monitor 其它 =====
WAIT_FEATURES_SECS      = 120
MONITOR_IDLE_TIMEOUT_S  = 120

# ===== 统一全局墙时长入口（仅此项允许运行期覆盖）=====
def _parse_int_like(x, default: int) -> int:
    try:
        return int(float(str(x)))
    except Exception:
        return default

# 默认值（未设环境变量时的兜底）；如需改动建议直接改此常量
DEFAULT_MAX_WALL_SECS: int = _parse_int_like(os.getenv("DEFAULT_MAX_WALL_SECS", None), -1)

# 模块导入快照：从环境变量读取一次（兼容旧逻辑），作为 get_max_wall_secs 的默认返回
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

# ===== 重训期间暂停/恢复控制（monitor<->infer）=====
MONITOR_SIGNAL_INFER_PAUSE = False  # monitor 触发重训时，是否给 infer 下发“暂停”旗标
INFER_RESPECT_PAUSE_FLAG   = True   # infer 是否响应暂停旗标
PAUSE_INFER_KEY            = f"{RESULT_DIR}/pause_infer.flag"
MONITOR_WAIT_RETRAIN       = False  # 保留旧名，避免外部引用报错（逻辑已内置）

# ===== Offline 训练控制（显式配置）=====
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

# ===== Retrain（显式配置）=====
RETRAIN_WARM_EPOCHS    = 3
RETRAIN_EPOCHS_FULL    = 8
RETRAIN_EARLY_PATIENCE = 2
RETRAIN_MODE           = "auto"   # "scratch" | "finetune" | "auto"
RETRAIN_FREEZE_N       = 0
RETRAIN_VAL_FRAC       = 0.2

# ===== 动态重训网格（显式配置）=====
def abc_grids(current_hidden: List[int] | None = None) -> Dict[str, Dict]:
    def uniq_layers(cands: List[List[int]]) -> List[List[int]]:
        seen = set(); out = []
        for h in cands:
            k = tuple(h)
            if k in seen:
                continue
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
            "activation":    ["relu", "mse"],
            "loss":          ["smoothl1", "mse"],
            "wd":            [0.0, 1e-4],
            "topk":          3,
        },
    }
