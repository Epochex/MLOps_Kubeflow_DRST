#!/usr/bin/env python3
"""
ml.train_offline.py - baseline MLP with PCA (bridge only)
────────────────────────────────────────────────────────────
• 若 TRIGGER = 0，跳过训练并下载已有 artefacts
• 若 TRIGGER = 1，从 datasets_old/old_total.csv 训练
• 仅使用 bridge 行数据
• PCA 保留 ≥85% 方差
• 输出 scaler / pca / model.pt / 预测结果等
"""

TRIGGER = 0  # 1 = 训练；0 = 跳过
import io
import sys, os, json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Dict, Tuple

from shared.minio_helper import load_csv, load_np, save_np, save_bytes, s3
from shared.config import MODEL_DIR, RESULT_DIR, TARGET_COL, BUCKET, DATA_DIR
from shared.metric_logger import log_metric
from shared.features import FEATURE_COLS
from shared.utils import calculate_accuracy_within_threshold
import shutil
from ml.model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

if TRIGGER == 0:
    """
    快速路径：跳过离线训练，仅下载已有 artefacts，并在 bridge/linear/dag1 上做准确率检查。
    """
    import shutil
    from pathlib import Path
    import joblib, torch, numpy as np

    # 目录准备
    model_dir_local  = Path("/mnt/pvc/models")
    result_dir_local = Path(f"/mnt/pvc/{RESULT_DIR}")
    model_dir_local.mkdir(parents=True, exist_ok=True)
    result_dir_local.mkdir(parents=True, exist_ok=True)

    # 1) 下载 artefacts
    artefacts = ["scaler.pkl", "pca.pkl", "baseline_model.pt", "model.pt"]
    for fname in artefacts:
        key = f"{MODEL_DIR}/{fname}"
        print(f"[offline][SKIP] downloading {key} …")
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        local_path = model_dir_local / fname
        with open(local_path, "wb") as f:
            f.write(obj["Body"].read())
        print(f"[offline][SKIP] saved → {local_path}")

    # 2) adaptive 初始 = baseline
    shutil.copy(model_dir_local / "baseline_model.pt",
                model_dir_local / "model.pt")
    print("[offline][SKIP] copied baseline_model.pt → model.pt")
    
    with open(model_dir_local / "model.pt", "rb") as f:
        save_bytes(f"{MODEL_DIR}/model.pt",
                f.read(),
                "application/octet-stream")
    print("[offline][SKIP] uploaded model.pt to MinIO (baseline clone)")

    # 3) 在 bridge/linear/dag1 上检查准确率
    try:
        # 加载预处理器和模型
        scaler = joblib.load(model_dir_local / "scaler.pkl")
        pca    = joblib.load(model_dir_local / "pca.pkl")
        baseline_model = torch.load(
            model_dir_local / "baseline_model.pt",
            map_location=device
        ).eval()

        # 加载并清洗数据集
        df_bridge = (load_csv(f"{DATA_DIR}/old_total.csv")
                     .replace({'<not counted>': np.nan}).dropna())
        df_linear = (load_csv(f"{DATA_DIR}/old_linear.csv")
                     .replace({'<not counted>': np.nan}).dropna())
        df_dag1   = (load_csv(f"{DATA_DIR}/old_dag-1.csv")
                     .replace({'<not counted>': np.nan}).dropna())

        # 丢弃不需要的列并重排
        for df in (df_bridge, df_linear, df_dag1):
            df.drop(columns=["input_rate", "latency"], errors="ignore", inplace=True)
        df_bridge = df_bridge.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)
        df_linear = df_linear.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)
        df_dag1   = df_dag1.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

        # 打印样本数
        print(f"[offline] bridge={len(df_bridge)}  linear={len(df_linear)}  dag1={len(df_dag1)}")

        # 定义一个评估函数
        def eval_acc(df, name):
            X_raw = df[FEATURE_COLS].astype(np.float32).values
            y_true = df[TARGET_COL].astype(np.float32).values
            X_pca = pca.transform(scaler.transform(X_raw)).astype(np.float32)
            with torch.no_grad():
                y_pred = baseline_model(
                    torch.from_numpy(X_pca).to(device)
                ).cpu().numpy()
            acc = calculate_accuracy_within_threshold(y_true, y_pred, 0.15)
            print(f"[offline] {name}-test accuracy = {acc:.2f}%")
            log_metric(component="offline",
                       event=f"{name}_accuracy",
                       accuracy=round(acc, 2))

        # 评估 bridge 和 dag1
        eval_acc(df_bridge, "bridge")
        eval_acc(df_dag1,   "dag-1")

    except Exception as e:
        print(f"[offline][SKIP] accuracy check failed: {e}")
        
    # ---------- 额外生成 bridge_true.npy / bridge_pred.npy ----------
    try:
        print("[offline][SKIP] dumping bridge artefacts …")
        Xb_raw = df_bridge[FEATURE_COLS].astype(np.float32).values
        yb_true = df_bridge[TARGET_COL].astype(np.float32).values
        Xb_pca  = pca.transform(scaler.transform(Xb_raw)).astype(np.float32)

        with torch.no_grad():
            yb_pred = baseline_model(
                torch.from_numpy(Xb_pca).to(device)
            ).cpu().numpy()

        # 本地 PVC
        os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)
        np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy", yb_true)
        np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy", yb_pred)

        # 上传 MinIO
        for fn, arr in (("bridge_true.npy", yb_true),
                        ("bridge_pred.npy", yb_pred)):
            buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
            save_bytes(f"{RESULT_DIR}/{fn}", buf.read(), "application/npy")

        print("[offline][SKIP] bridge artefacts generated & uploaded")
    except Exception as e:
        print(f"[offline][SKIP] ❌ failed to dump bridge artefacts: {e}")

        

    # 4) 记录并退出
    log_metric(component="offline", event="skip_train")
    os.makedirs("/tmp/kfp_outputs", exist_ok=True)
    open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")

    print("[offline][SKIP] artefacts ready, training skipped – bye.")
    sys.exit(0)
# ----------------------------------------------------------------------

# ─── 1. 读取 & 清洗 ──────────────────────────────────────────────
def _read_clean(rel_path: str) -> pd.DataFrame:          # <<< 修改：提炼公共函数
    df = (load_csv(rel_path)
          .replace({'<not counted>': np.nan})
          .replace(r'^\s*$', np.nan, regex=True)
          .dropna())
    df.drop(columns=["input_rate", "latency"], errors="ignore", inplace=True)
    return df.reindex(columns=FEATURE_COLS + [TARGET_COL], fill_value=0.0)

df_bridge = _read_clean(f"{DATA_DIR}/old_total.csv")     # <<< 加 f
df_linear = _read_clean(f"{DATA_DIR}/old_linear.csv")    # <<< 恢复 linear
df_dag1   = _read_clean(f"{DATA_DIR}/old_dag-1.csv")

print(f"[offline] bridge={len(df_bridge)}  linear={len(df_linear)}  dag1={len(df_dag1)}")

# —— 仅 bridge 用来训练；linear 只是备用 / 对比 ——
Xb = df_bridge[FEATURE_COLS].astype(np.float32).values
yb = df_bridge[TARGET_COL].astype(np.float32).values
X1 = df_dag1 [FEATURE_COLS].astype(np.float32).values     # dag-1 用于后验评估
y1 = df_dag1 [TARGET_COL].astype(np.float32).values

# ─── 2. 标准化 ──────────────────────────────────────────────────
scaler = StandardScaler().fit(Xb)
Xb_s   = scaler.transform(Xb)
X1_s   = scaler.transform(X1)

# ─── 3. PCA（保留 ≥85% 方差）────────────────────────────────────
cumvar   = np.cumsum(PCA().fit(Xb_s).explained_variance_ratio_)
n_comp   = int(np.searchsorted(cumvar, 0.85) + 1)
pca      = PCA(n_components=n_comp).fit(Xb_s)

Xb_p = pca.transform(Xb_s).astype(np.float32)
X1_p = pca.transform(X1_s).astype(np.float32)

# ─── 4. Train / Test Split ─────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    Xb_p, yb, test_size=0.3, random_state=0
)

# ─── 5. DataLoader & 模型定义 ───────────────────────────────────
bs = 16
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
    batch_size=bs, shuffle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"


model = build_model(
    {"hidden_layers": (64, 32), "activation": "relu"},
    input_dim=n_comp
).to(device)

opt    = Adam(model.parameters(), lr=1e-2)
sched  = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
lossf  = nn.SmoothL1Loss()

# ─── 6. 训练（Early-Stop，patience=10）──────────────────────────
best_val, best_state, no_imp = float("inf"), None, 0
for epoch in range(1, 51):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        lossf(model(xb), yb).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = float(lossf(model(torch.from_numpy(X_te).to(device)),
                               torch.from_numpy(y_te).to(device)))
    sched.step(val_loss)

    if val_loss + 1e-6 < best_val:
        best_val, best_state, no_imp = val_loss, model.state_dict(), 0
    else:
        no_imp += 1
        if no_imp >= 10:
            print(f"[offline] early-stop @ epoch {epoch}")
            break

model.load_state_dict(best_state)

# ─── 7. 评估 ────────────────────────────────────────────────────
with torch.no_grad():
    y_pred_te = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
acc_te = calculate_accuracy_within_threshold(y_te, y_pred_te, 0.15)
print(f"[offline] bridge-test accuracy = {acc_te:.2f}%")

with torch.no_grad():
    y_pred_d1 = model(torch.from_numpy(X1_p).to(device)).cpu().numpy()
acc_d1 = calculate_accuracy_within_threshold(y1, y_pred_d1, 0.15)
print(f"[offline] dag-1 accuracy       = {acc_d1:.2f}%")

# ─── 8. 保存 artefacts ─────────────────────────────────────────
# <<< 一定要在 dump 之前创建目录，否则 FileNotFoundError >>>
os.makedirs("/mnt/pvc/models", exist_ok=True)
os.makedirs(f"/mnt/pvc/{RESULT_DIR}", exist_ok=True)

# 1) 本地持久化：scaler、pca、baseline_model、model
joblib.dump(scaler,            "/mnt/pvc/models/scaler.pkl")
joblib.dump(pca,               "/mnt/pvc/models/pca.pkl")
torch.save(model.eval().cpu(), "/mnt/pvc/models/baseline_model.pt")
torch.save(model.eval().cpu(), "/mnt/pvc/models/model.pt")

# 2) 保存预测结果
np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_true.npy",  y_te)
np.save(f"/mnt/pvc/{RESULT_DIR}/bridge_pred.npy",  y_pred_te)

# 3) 上传到 MinIO
for fn in ("scaler.pkl", "pca.pkl", "baseline_model.pt", "model.pt"):
    with open(f"/mnt/pvc/models/{fn}", "rb") as f:
        save_bytes(f"{MODEL_DIR}/{fn}", f.read(), "application/octet-stream")

for arr in ("bridge_true.npy", "bridge_pred.npy"):
    with open(f"/mnt/pvc/{RESULT_DIR}/{arr}", "rb") as f:
        save_bytes(f"{RESULT_DIR}/{arr}", f.read(), "application/npy")

# 4) 上传时间戳
save_bytes(f"{MODEL_DIR}/last_update_utc.txt",
           datetime.utcnow().isoformat().encode(), "text/plain")

print("[offline] artefacts uploaded ✔")

# ─── 9. KFP metadata ───────────────────────────────────────────
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
