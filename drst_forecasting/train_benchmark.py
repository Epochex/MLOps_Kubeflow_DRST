# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/train_benchmark.py
from __future__ import annotations
import os, io, json, time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from drst_common.config import MODEL_DIR, RESULT_DIR, ACC_THR
from drst_common.minio_helper import save_bytes
from drst_common.metric_logger import log_metric
from drst_common.resource_probe import start as start_probe

from .dataset import build_sliding_window
from .models import LSTMForecaster
from .metrics import mae, rmse, acc_within_threshold
from .baseline_mean import moving_mean_baseline

# ====== 可调参数（也可由 pipeline 组件通过环境变量传入） ======
LOOKBACK  = int(os.getenv("FORECAST_LOOKBACK", "48"))   # 历史窗口长度
HORIZON   = int(os.getenv("FORECAST_HORIZON",  "12"))   # 预测步数
BATCH     = int(os.getenv("FORECAST_BS",       "64"))
LR        = float(os.getenv("FORECAST_LR",     "1e-3"))
EPOCHS    = int(os.getenv("FORECAST_EPOCHS",   "20"))
PATIENCE  = int(os.getenv("FORECAST_PATIENCE", "5"))
HIDDEN    = int(os.getenv("FORECAST_HIDDEN",   "64"))
LAYERS    = int(os.getenv("FORECAST_LAYERS",   "1"))
TAKE_LAST = int(os.getenv("FORECAST_TAKE_LAST","0"))  # 0 表示不裁剪

device = "cuda" if torch.cuda.is_available() else "cpu"

def _train_once(X, Y) -> Tuple[torch.nn.Module, dict]:
    # 7:3 划分
    N = len(X)
    n_va = max(1, int(N * 0.3))
    Xtr, Xva = X[:-n_va], X[-n_va:]
    Ytr, Yva = Y[:-n_va], Y[-n_va:]

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    model = LSTMForecaster(in_dim=X.shape[-1], hidden=HIDDEN, num_layers=LAYERS, horizon=HORIZON).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.SmoothL1Loss()

    best_rmse = float("inf"); best_state: bytes | None = None; bad = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward(); opt.step()

        # 验证
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
        cur_rmse = rmse(Yva, pv)

        if cur_rmse < best_rmse - 1e-6:
            best_rmse = cur_rmse; bad = 0
            bio = io.BytesIO(); torch.save(model.to("cpu"), bio); best_state = bio.getvalue(); model.to(device)
        else:
            bad += 1

        if ep == 1 or ep % 5 == 0:
            print(f"[forecast.train] ep={ep:03d}/{EPOCHS} rmse={cur_rmse:.5f} best={best_rmse:.5f}", flush=True)
        if bad >= PATIENCE:
            print(f"[forecast.train] early-stop at {ep} (no improve {bad})", flush=True)
            break

    if best_state is not None:
        model = torch.load(io.BytesIO(best_state), map_location=device)

    # 最终评估
    with torch.no_grad():
        pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy()
    m_mae  = mae(Yva, pv)
    m_rmse = rmse(Yva, pv)
    m_acc  = acc_within_threshold(Yva, pv, thr=float(ACC_THR))

    # baseline
    # 取每个样本的**目标历史窗口**：由于 X 使用的是特征（多变量），baseline 用 Y 的历史需要额外准备；
    # 这里简化：直接用“最后一个训练样本的目标历史均值”作为估计（常见课题基线写法）
    y_hist_for_last = Ytr[:, :1] * 0.0  # 占位避免误导（真实项目可在构造窗口时单独保留目标历史滑窗）
    base_pred = np.full_like(pv, fill_value=float(np.mean(Ytr)))  # 粗基线：全局均值
    base_mae  = mae(Yva, base_pred)
    base_rmse = rmse(Yva, base_pred)
    base_acc  = acc_within_threshold(Yva, base_pred, thr=float(ACC_THR))

    metrics = {
        "mae": m_mae, "rmse": m_rmse, f"acc@{ACC_THR:.2f}".rstrip("0").rstrip("."): m_acc,
        "baseline_mae": base_mae, "baseline_rmse": base_rmse, "baseline_acc": base_acc,
        "lookback": LOOKBACK, "horizon": HORIZON, "hidden": HIDDEN, "layers": LAYERS,
        "epochs": EPOCHS, "bs": BATCH, "lr": LR, "patience": PATIENCE,
    }
    return model, metrics

def main():
    stop_probe = start_probe("forecast_train")
    t0 = time.time()
    print(f"[forecast.train] start lookback={LOOKBACK} horizon={HORIZON}", flush=True)

    X, Y, feats = build_sliding_window(LOOKBACK, HORIZON, take_last_n=(None if TAKE_LAST <= 0 else TAKE_LAST))
    model, metrics = _train_once(X, Y)

    # 保存模型
    buf = io.BytesIO(); torch.save(model.to("cpu"), buf)
    save_bytes(f"{MODEL_DIR}/forecasting/model.pt", buf.getvalue(), "application/octet-stream")

    # 保存验证集最后一个样本的预测（便于 plot/serve 对齐）
    with torch.no_grad():
        pv = model(torch.from_numpy(X[-1:]).float().to(device)).cpu().numpy()[0]  # [H]
    np.save(io.BytesIO(), pv)
    bio = io.BytesIO(); np.save(bio, pv); bio.seek(0)
    save_bytes(f"{RESULT_DIR}/forecasting/pred_last_window.npy", bio.getvalue(), "application/npy")

    # 记录指标
    ts = int(time.time())
    save_bytes(f"{RESULT_DIR}/forecasting/metrics_{ts}.json", json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
    log_metric(component="forecasting", event="train_done", wall_s=round(time.time()-t0,3),
               mae=metrics["mae"], rmse=metrics["rmse"], acc=metrics.get(f"acc@{ACC_THR:.2f}".rstrip('0').rstrip('.'), None))
    print(f"[forecast.train] done rmse={metrics['rmse']:.5f} mae={metrics['mae']:.5f}", flush=True)
    stop_probe()

if __name__ == "__main__":
    main()
