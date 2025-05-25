"""
离线初训：4 个 CSV → 标准化 → PCA → 训练基线 MLP
保存：
  models/
     ├─ scaler.pkl
     ├─ pca.pkl
     ├─ mlp.pt
     └─ base_acc.npy   ← ★ 新增
"""
import numpy as np, pandas as pd, pathlib, joblib, os, torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from ml.model        import train_mlp
from shared.config   import DATA_DIR, MODEL_DIR, TARGET_COL
from shared.features import FEATURE_COLS
from shared.utils    import calculate_accuracy_within_threshold

from shared.minio_helper import load_csv, save_bytes

def load_csv(rel_path):
    return (pd.read_csv(rel_path, index_col=0)
              .replace({'<not counted>': np.nan, ' ': np.nan})
              .dropna())

from shared.config   import TARGET_COL

# ---------- 读 Bridge CSV ----------
bridge = load_csv(f"{DATA_DIR}/total/resource_stimulus_global_A-B-C_modified.csv")

# ---------- 标准化 + PCA ----------
scaler = StandardScaler().fit(bridge[FEATURE_COLS])
Xp     = scaler.transform(bridge[FEATURE_COLS])

pca_full = PCA().fit(Xp)
n_components = np.where(np.cumsum(
        pca_full.explained_variance_ratio_) >= .85)[0][0] + 1
pca  = PCA(n_components=n_components).fit(Xp)
Xp   = pca.transform(Xp).astype(np.float32)
y    = bridge[TARGET_COL].values.astype(np.float32)

# ---------- 训练基线 MLP ----------
X_tr, X_val, y_tr, y_val = train_test_split(
        Xp, y, test_size=.3, random_state=40)
model, y_pred_val = train_mlp(X_tr, y_tr, X_val, y_val)

base_acc = calculate_accuracy_within_threshold(y_val, y_pred_val)
print(f"[offline] val acc={base_acc:.2f}%  PCA n_components={n_components}")

# ---------- 保存 artefacts ----------
# ---------- 保存 artefacts 到 MinIO ----------
save_pkl(f"{MODEL_DIR}/scaler.pkl", scaler)
save_pkl(f"{MODEL_DIR}/pca.pkl",    pca)

buf = pathlib.Path("mlp.tmp")
torch.save(model.state_dict(), buf)
save_bytes(f"{MODEL_DIR}/mlp.pt", buf.read_bytes())
buf.unlink()
save_np(f"{MODEL_DIR}/base_acc.npy", np.array(base_acc))
