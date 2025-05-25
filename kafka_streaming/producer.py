import json, time, pathlib, sys, os
import pandas as pd, numpy as np
from kafka import KafkaProducer
from shared.features import FEATURE_COLS
from shared.config   import (DATA_DIR, KAFKA_TOPIC, KAFKA_SERVERS,
                             TARGET_COL, BATCH_SIZE)




df = load_csv(f"{DATA_DIR}/dag-1/resource_stimulus_global_A-B-C.csv")
producer = KafkaProducer(bootstrap_servers=",".join(KAFKA_SERVERS))

total = len(df); sent = 0
for idx, row in df.iterrows():
    msg = {
        "row_index": int(idx),
        "features": {c: float(row[c]) for c in FEATURE_COLS},
        "label":    float(row[TARGET_COL])
    }
    producer.send(KAFKA_TOPIC, json.dumps(msg).encode())
    sent += 1
    if sent % BATCH_SIZE == 0 or sent == total:
        producer.flush()
        print(f"[producer] batch {sent//BATCH_SIZE:3d} | sent {sent}/{total}")
    time.sleep(0.05)
producer.close()
print("[producer] DONE")
