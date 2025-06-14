# shared/features.py
# ------------------------------------------------------------
# “唯一真理源”：old_total.csv 决定 60 个特征列
# ------------------------------------------------------------
import numpy as np
from .config         import DATA_DIR, TARGET_COL, EXCLUDE_COLS
from .minio_helper   import load_csv

# 直接以旧版全量数据作为基准
SRC_KEY  = f"{DATA_DIR}/old_total.csv"

_df_src  = (
    load_csv(SRC_KEY)
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
