"""
持续消费 Kafka：
1. 每 batch(60 条) 监测精度 + (可选) GridSearch 细粒度重训
2. Producer 结束、空闲超过 IDLE_TIMEOUT 秒 ⇒ 退出循环
3. 汇总整段 Dag-1 样本，按原 notebook Section 7-8 逻辑生成
   report_final.png（0-200 index、Phase1-3）。

可 Ctrl-C 强停。
"""
import json, time, threading, queue, joblib, pathlib, os, sys, warnings
import numpy as np, torch, pandas as pd
from kafka import KafkaConsumer
from shared.config   import (KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE,
                             MODEL_DIR, TARGET_COL, DATA_DIR)
from shared.features import FEATURE_COLS
from shared.utils    import calculate_accuracy_within_threshold
from ml.model        import MLP
from ml.drift_detector import gridsearch_retrain
from ml.plot_report  import generate_report

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 载离线 artefacts ----------
scaler  = joblib.load(pathlib.Path(MODEL_DIR, "scaler.pkl"))
pca     = joblib.load(pathlib.Path(MODEL_DIR, "pca.pkl"))
in_dim  = pca.n_components_
model   = MLP(in_dim)
model.load_state_dict(torch.load(pathlib.Path(MODEL_DIR, "mlp.pt"),
                                 map_location="cpu"))
model.eval()

BASE_ACC = np.load(pathlib.Path(MODEL_DIR, "base_acc.npy"))

# ---------- 提前准备 Bridge 预测（Phase-1 左段用） ----------
path_bridge = pathlib.Path(DATA_DIR, "total",
                           "resource_stimulus_global_A-B-C_modified.csv")
df_bridge = (pd.read_csv(path_bridge, index_col=0)
               .replace({'<not counted>': np.nan, ' ': np.nan})
               .dropna())
Xb_raw = df_bridge[FEATURE_COLS].values.astype(np.float32)
y_bridge_true = df_bridge[TARGET_COL].values.astype(np.float32)
Xb_pca = pca.transform(scaler.transform(Xb_raw)).astype(np.float32)
with torch.no_grad():
    bridge_pred_orig = model(torch.from_numpy(Xb_pca)).cpu().numpy()

# ---------- kafka listener ----------
rows_q  = queue.Queue()
stop_ev = threading.Event()

def listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        auto_offset_reset="latest",        # 仅读取启动后新消息
        value_deserializer=lambda m: json.loads(m.decode()))
    for msg in cons:
        if stop_ev.is_set():
            break
        rows_q.put(msg.value)
    cons.close()

threading.Thread(target=listener, daemon=True).start()

IDLE_TIMEOUT = 30   # s
batch_id     = 0
all_rows     = []   # 汇总整段 Dag-1

try:
    while True:
        # --------- 取一批 60 条 ----------
        batch = []
        t0 = time.time()
        while len(batch) < BATCH_SIZE and not stop_ev.is_set():
            try:
                batch.append(rows_q.get(timeout=1))
            except queue.Empty:
                if time.time() - t0 > IDLE_TIMEOUT:
                    break
        if not batch:
            print("[consumer] idle timeout, exit loop")
            break

        batch_id += 1
        all_rows.extend(batch)

        # --------- 推断 + 精度 ----------
        X_raw = np.array([[r["features"][c] for c in FEATURE_COLS] for r in batch],
                         dtype=np.float32)
        y_true = np.array([r["label"] for r in batch], dtype=np.float32)

        Xp = pca.transform(scaler.transform(X_raw)).astype(np.float32)
        with torch.no_grad():
            y_pred = model(torch.from_numpy(Xp)).cpu().numpy()

        acc = calculate_accuracy_within_threshold(y_true, y_pred)
        print(f"[consumer] batch {batch_id:3d} | acc={acc:.2f}% "
              f"(base {BASE_ACC:.2f})")

        # --------- 漂移判定 & 可选 GridSearch ----------
        if BASE_ACC - acc > 5:
            print("   ↳ drift detected, GridSearch retrain …")
            model = gridsearch_retrain(Xp, y_true, model, in_dim)
            torch.save(model.state_dict(), pathlib.Path(MODEL_DIR,"mlp_latest.pt"))
            print("   ↳ new model saved")

except KeyboardInterrupt:
    print("\n[consumer] Ctrl-C pressed, stopping…")
finally:
    stop_ev.set()
    time.sleep(1)

# =========================================================
# 汇总整段 Dag-1，生成终极报告
# =========================================================
if not all_rows:
    print("[consumer] no rows collected, skip report")
    sys.exit(0)

df_dag1 = pd.DataFrame(
    [{**r["features"], TARGET_COL: r["label"]} for r in all_rows])
Xd1_raw = df_dag1[FEATURE_COLS].values.astype(np.float32)
yd1_true= df_dag1[TARGET_COL].values.astype(np.float32)
Xd1_pca = pca.transform(scaler.transform(Xd1_raw)).astype(np.float32)

with torch.no_grad():
    dag1_pred_orig = MLP(in_dim)
    dag1_pred_orig.load_state_dict(torch.load(
        pathlib.Path(MODEL_DIR, "mlp.pt"), map_location="cpu"))
    dag1_pred_orig.eval()
    dag1_pred_orig = dag1_pred_orig(torch.from_numpy(Xd1_pca)).numpy()

with torch.no_grad():
    y_pred_new = model(torch.from_numpy(Xd1_pca)).cpu().numpy()

print("[consumer] ---- generating final 0-200 phase report ----")
generate_report(
    bridge_true        = y_bridge_true,
    bridge_pred_orig   = bridge_pred_orig,
    dag1_pred_orig     = dag1_pred_orig,
    y_pred_dag1_new    = y_pred_new,
    yd1                = yd1_true,
    correction_offset  = 38,
    save_path="results/report_final.png"
)

print("[consumer] DONE.  see results/report_final.png")
