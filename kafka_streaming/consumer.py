import json, time, threading, queue, io, warnings, pathlib, sys
import numpy as np, torch, pandas as pd
from kafka import KafkaConsumer
from shared.minio_helper import (load_csv, save_bytes, save_np, load_np,
                                 save_pkl, s3, BUCKET, MODEL_DIR, DATA_DIR, RESULT_DIR)
from shared.config   import (KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE,
                             CONSUME_IDLE_S, CORRECTION_OFFSET, TARGET_COL)
from shared.features import FEATURE_COLS
from shared.utils    import calculate_accuracy_within_threshold
from ml.model        import MLP
from ml.drift_detector import gridsearch_retrain
from ml.plot_report  import generate_report
import joblib, io

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 载 artefacts ----------
scaler = joblib.load(io.BytesIO(s3.get_object(
        Bucket=BUCKET, Key=f"{MODEL_DIR}/scaler.pkl")["Body"].read()))
pca    = joblib.load(io.BytesIO(s3.get_object(
        Bucket=BUCKET, Key=f"{MODEL_DIR}/pca.pkl")["Body"].read()))
in_dim = pca.n_components_
state  = io.BytesIO(s3.get_object(
        Bucket=BUCKET, Key=f"{MODEL_DIR}/mlp.pt")["Body"].read())
model  = MLP(in_dim); model.load_state_dict(torch.load(state)); model.eval()
BASE_ACC = float(load_np(f"{MODEL_DIR}/base_acc.npy"))

# ---------- Bridge 预测 ----------
bridge = load_csv(f"{DATA_DIR}/total/resource_stimulus_global_A-B-C_modified.csv")
Xb_pca = pca.transform(scaler.transform(bridge[FEATURE_COLS])).astype(np.float32)
with torch.no_grad():
    bridge_pred_orig = model(torch.from_numpy(Xb_pca)).numpy()
yb_bridge = bridge[TARGET_COL].values.astype(np.float32)

# ---------- Kafka 监听 ----------
rows_q, stop_ev = queue.Queue(), threading.Event()

def listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC, bootstrap_servers=KAFKA_SERVERS,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode()))
    for msg in cons:
        if stop_ev.is_set(): break
        rows_q.put(msg.value)
    cons.close()

threading.Thread(target=listener, daemon=True).start()

def take_batch(size):
    rows, t0 = [], time.time()
    while len(rows)<size and time.time()-t0<CONSUME_IDLE_S:
        try: rows.append(rows_q.get(timeout=1))
        except queue.Empty: pass
    return rows

all_rows=[]; batch_id=0
while True:
    batch = take_batch(BATCH_SIZE)
    if not batch:
        print("[consumer] idle timeout, exit loop"); break
    batch_id+=1; all_rows.extend(batch)

    X = np.array([[r["features"][c] for c in FEATURE_COLS] for r in batch],
                 dtype=np.float32)
    y = np.array([r["label"] for r in batch], dtype=np.float32)
    Xp = pca.transform(scaler.transform(X)).astype(np.float32)
    with torch.no_grad(): y_pred = model(torch.from_numpy(Xp)).numpy()
    acc = calculate_accuracy_within_threshold(y, y_pred)
    print(f"[consumer] batch {batch_id:3d} | acc={acc:.2f}% (base {BASE_ACC:.2f})")

    if BASE_ACC - acc > 5:
        print("   ↳ drift detected, retrain …")
        model = gridsearch_retrain(Xp, y, model, in_dim)
        buf=io.BytesIO(); torch.save(model.state_dict(),buf);buf.seek(0)
        save_bytes(f"{MODEL_DIR}/mlp_latest.pt", buf.read())

# ---------- 汇总结果 & 画最终图 ----------
df_dag1 = pd.DataFrame(
    [{**r["features"], TARGET_COL:r["label"]} for r in all_rows])
Xd_pca  = pca.transform(scaler.transform(df_dag1[FEATURE_COLS])).astype(np.float32)
yd1     = df_dag1[TARGET_COL].values.astype(np.float32)
with torch.no_grad():
    dag1_pred_orig = model(torch.from_numpy(Xd_pca)).numpy()
generate_report(
    bridge_true        = yb_bridge,
    bridge_pred_orig   = bridge_pred_orig,
    dag1_pred_orig     = dag1_pred_orig,
    y_pred_dag1_new    = dag1_pred_orig,
    yd1                = yd1,
    correction_offset  = CORRECTION_OFFSET,
    save_path          = "report_final.png")
# 上传
save_bytes(f"{RESULT_DIR}/report_final.png",
           open("report_final.png","rb").read(), "image/png")
print("[consumer] DONE & uploaded report_final.png")
