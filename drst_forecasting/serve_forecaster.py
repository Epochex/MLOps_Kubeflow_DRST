# /data/mlops/DRST-SoftwarizedNetworks/drst_forecasting/serve_forecaster.py
from __future__ import annotations
import os, io, json, time
import numpy as np
import torch

from drst_common.config import MODEL_DIR, RESULT_DIR, DATA_DIR, TARGET_COL
from drst_common.minio_helper import load_csv, save_bytes, s3, BUCKET
from drst_common.resource_probe import start as start_probe
from .models import LSTMForecaster
from .dataset import _load_selected_features, _load_offline_scaler

LOOKBACK = int(os.getenv("FORECAST_LOOKBACK", "48"))
HORIZON  = int(os.getenv("FORECAST_HORIZON",  "12"))

device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    raw = s3.get_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/forecasting/model.pt")["Body"].read()
    model = torch.load(io.BytesIO(raw), map_location=device)
    model.eval()
    return model

def main():
    stop = start_probe("forecast_serve")
    print("[forecast.serve] start", flush=True)

    feats = _load_selected_features()
    df = load_csv(f"{DATA_DIR}/combined.csv").reset_index(drop=True)
    for c in feats + [TARGET_COL]: df[c] = np.asarray(df[c], dtype=np.float32)
    df = df.dropna(subset=feats + [TARGET_COL]).reset_index(drop=True)

    sc = _load_offline_scaler()
    X_all = sc.transform(df[feats].astype(np.float32).values)   # [N,D]
    if len(X_all) < LOOKBACK:
        raise RuntimeError("数据长度不足以拿到最后一个 lookback 窗口")
    last = X_all[-LOOKBACK:, :]                                 # [T,D]

    model = _load_model()
    with torch.no_grad():
        pred = model(torch.from_numpy(last[None, ...]).float().to(device)).cpu().numpy()[0]  # [H]

    ts = int(time.time())
    # CSV
    import pandas as pd
    out = pd.DataFrame({"step": list(range(1, HORIZON+1)), "pred": pred.astype(np.float32)})
    bio = io.BytesIO(); out.to_csv(bio, index=False); save_bytes(f"{RESULT_DIR}/forecasting/forecast_{ts}.csv", bio.getvalue(), "text/csv")
    # JSON
    save_bytes(f"{RESULT_DIR}/forecasting/forecast_{ts}.json", json.dumps({"horizon": HORIZON, "pred": pred.tolist()}, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
    print(f"[forecast.serve] wrote forecast_{ts}.csv/json", flush=True)
    stop()

if __name__ == "__main__":
    main()
