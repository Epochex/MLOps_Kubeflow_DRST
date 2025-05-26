# kafka_streaming/consumer.py
import json, time, threading, queue, io, warnings
import numpy as np, torch, pandas as pd, joblib
from kafka import KafkaConsumer

from shared.minio_helper import load_csv, save_np, save_bytes, load_np, s3, BUCKET
from shared.config import (
    DATA_DIR, MODEL_DIR, RESULT_DIR,
    KAFKA_TOPIC, KAFKA_SERVERS,
    BATCH_SIZE, CONSUME_IDLE_S,
    CORRECTION_OFFSET, TARGET_COL
)
from shared.features import FEATURE_COLS
from shared.utils import calculate_accuracy_within_threshold
from ml.model import MLP
from ml.drift_detector import gridsearch_retrain

warnings.filterwarnings("ignore", category=UserWarning)

# ——— 1) 静态 Bridge 预测并写入 ———
scaler_buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")['Body'].read()
)
scaler = joblib.load(scaler_buf)
pca_buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/pca.pkl")['Body'].read()
)
pca = joblib.load(pca_buf)

state_buf = io.BytesIO(
    s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/mlp.pt")['Body'].read()
)
model = MLP(pca.n_components_)
model.load_state_dict(torch.load(state_buf))
model.eval()

BASE_ACC = float(load_np(f"{MODEL_DIR}/base_acc.npy"))

df_bridge = load_csv(f"{DATA_DIR}/total/resource_stimulus_global_A-B-C_modified.csv")
Xb = pca.transform(scaler.transform(df_bridge[FEATURE_COLS])).astype(np.float32)
with torch.no_grad():
    bridge_pred_orig = model(torch.from_numpy(Xb)).numpy()
bridge_true = df_bridge[TARGET_COL].values.astype(np.float32)

for key, arr in [
    (f"{RESULT_DIR}/bridge_true.npy", bridge_true),
    (f"{RESULT_DIR}/bridge_pred.npy", bridge_pred_orig),
]:
    print(f"[write] uploading to MinIO: {key}")
    try:
        save_np(key, arr)
    except Exception as e:
        print(f"[error] failed to write {key}: {e}")

# ——— 2) 有限 Kafka 流式消费 ———
rows_q, stop_ev = queue.Queue(), threading.Event()
def listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode())
    )
    for msg in cons:
        if stop_ev.is_set(): break
        rows_q.put(msg.value)
    cons.close()
threading.Thread(target=listener, daemon=True).start()

def take_batch(n):
    buf, t0 = [], time.time()
    while len(buf) < n and time.time() - t0 < CONSUME_IDLE_S:
        try: buf.append(rows_q.get(timeout=1))
        except queue.Empty: pass
    return buf

all_rows = []
while True:
    batch = take_batch(BATCH_SIZE)
    if not batch:
        print("[consumer] idle timeout, stop streaming")
        break
    all_rows.extend(batch)
    X_raw = np.array([[r["features"][c] for c in FEATURE_COLS] for r in batch], dtype=np.float32)
    y_true = np.array([r["label"] for r in batch], dtype=np.float32)
    Xp = pca.transform(scaler.transform(X_raw)).astype(np.float32)
    with torch.no_grad():
        y_pred_orig = model(torch.from_numpy(Xp)).numpy()
    acc = calculate_accuracy_within_threshold(y_true, y_pred_orig)
    print(f"[consumer] batch acc={acc:.2f}% (base {BASE_ACC:.2f}%)")
    if BASE_ACC - acc > 5:
        print("[consumer] drift detected, retraining")
        model = gridsearch_retrain(Xp, y_true, model, pca.n_components_)
        buf = io.BytesIO(); torch.save(model.state_dict(), buf); buf.seek(0)
        key = f"{MODEL_DIR}/mlp_latest.pt"
        print(f"[write] uploading to MinIO: {key}")
        try:
            save_bytes(key, buf.read())
        except Exception as e:
            print(f"[error] failed to write {key}: {e}")

# ——— 3) 汇总 Dag-1 并保证写入 npy ———
if all_rows:
    df_dag1 = pd.DataFrame([{**r["features"], TARGET_COL: r["label"]} for r in all_rows])
    Xd1 = pca.transform(scaler.transform(df_dag1[FEATURE_COLS])).astype(np.float32)
    y_true_dag1 = df_dag1[TARGET_COL].values.astype(np.float32)
    with torch.no_grad():
        dag1_pred_orig = model(torch.from_numpy(Xd1)).numpy()
    idx_arr = np.arange(len(y_true_dag1), dtype=np.int32)
else:
    print("[consumer] no streaming data, writing empty arrays")
    dag1_pred_orig = np.empty((0,), dtype=np.float32)
    y_true_dag1    = np.empty((0,), dtype=np.float32)
    idx_arr        = np.empty((0,), dtype=np.int32)

for key, arr in [
    (f"{RESULT_DIR}/dag1_pred_orig.npy", dag1_pred_orig),
    (f"{RESULT_DIR}/dag1_true.npy",      y_true_dag1),
    (f"{RESULT_DIR}/idx.npy",            idx_arr),
]:
    print(f"[write] uploading to MinIO: {key}")
    try:
        save_np(key, arr)
    except Exception as e:
        print(f"[error] failed to write {key}: {e}")

print("[consumer] DONE")
