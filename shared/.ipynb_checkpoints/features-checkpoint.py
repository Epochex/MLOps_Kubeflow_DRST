import pandas as pd, numpy as np, pathlib, json, os
from .config import DATA_DIR, EXCLUDE_COLS, TARGET_COL

csv_bridge = pathlib.Path(DATA_DIR, "total",
                          "resource_stimulus_global_A-B-C_modified.csv")
df_bridge  = (pd.read_csv(csv_bridge, index_col=0)
                .replace({'<not counted>': np.nan, ' ': np.nan})
                .dropna())

FEATURE_COLS = [c for c in df_bridge.columns
                if c not in EXCLUDE_COLS + [TARGET_COL]]

# 供 Producer & Consumer import
if __name__ == "__main__":
    print(json.dumps(FEATURE_COLS, indent=2))
