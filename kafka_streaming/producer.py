import json, time, pathlib, sys, os
import pandas as pd, numpy as np
from kafka import KafkaProducer
from shared.features import FEATURE_COLS
from shared.config   import (DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
                             TARGET_COL, BATCH_SIZE)

CSV = pathlib.Path(DATA_DIR, "dag-1",
                   "resource_stimulus_global_A-B-C.csv")
df  = (pd.read_csv(CSV, index_col=0)
         .replace({'<not counted>': np.nan, ' ': np.nan})
         .dropna()
         .reset_index(drop=True))

def run_producer():
    prod  = KafkaProducer(bootstrap_servers=",".join(KAFKA_SERVERS))
    total = len(df)
    sent  = 0
    for idx, row in df.iterrows():
        msg = {
            "row_index": int(idx),
            "features": {c: float(row[c]) for c in FEATURE_COLS},
            "label":    float(row[TARGET_COL])
        }
        prod.send(KAFKA_TOPIC, json.dumps(msg).encode())
        sent += 1
        if sent % BATCH_SIZE == 0 or sent == total:
            prod.flush()
            print(f"[producer] batch {sent//BATCH_SIZE:3d} | "
                  f"sent {sent}/{total}")
        time.sleep(0.05)            # 20 TPS
    prod.close()
    print("[producer] DONE")

if __name__ == "__main__":
    run_producer()
