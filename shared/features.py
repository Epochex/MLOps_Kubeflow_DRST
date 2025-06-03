# shared/features.py
import pandas as pd, numpy as np, pathlib
from .config import DATA_DIR, EXCLUDE_COLS, TARGET_COL
from .minio_helper import load_csv          # ← 用 MinIO 工具

# -------- 1. 正确文件名 -------------
BRIDGE_KEY = f"{DATA_DIR}/combined.csv"     # ✅ 与 MinIO 中保持一致

# -------- 2. 直接 MinIO 读取 ---------
df_bridge = (
    load_csv(BRIDGE_KEY)                    # ← MinIO 拉取
      .replace({'<not counted>': np.nan})
      .dropna()
)

FEATURE_COLS = [
    c for c in df_bridge.columns
    if c not in EXCLUDE_COLS + [TARGET_COL]
]

if __name__ == "__main__":
    from pprint import pprint
    pprint(FEATURE_COLS)
