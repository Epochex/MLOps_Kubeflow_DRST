# shared/features.py
# 唯一“真理源”：datasets/random_rates.csv 的 60 个特征列
import numpy as np, pandas as pd
from .config         import DATA_DIR, TARGET_COL, EXCLUDE_COLS
from .minio_helper   import load_csv     # ➜ 自动支持本地 / MinIO
import os, io, json

SRC_KEY  = f"{DATA_DIR}/random_rates.csv"    # Producer 用的同一文件
_df_src  = (
    load_csv(SRC_KEY)                        # 先本地，再 MinIO
      .replace({'<not counted>': np.nan})
      .dropna()
)

FEATURE_COLS = [
    c for c in _df_src.columns
    if c not in EXCLUDE_COLS + [TARGET_COL]
]

# —— 自检 ——  
if len(FEATURE_COLS) != 60:
    raise ValueError(f"[features] expect 60 cols, got {len(FEATURE_COLS)}")

if __name__ == "__main__":
    from pprint import pprint
    pprint(FEATURE_COLS)
