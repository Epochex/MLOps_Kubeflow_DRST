#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# drst_forecasting/publish_forecast.py

from __future__ import annotations
import io, json, time
import numpy as np
import pandas as pd

from drst_common.minio_helper import s3, save_bytes, BUCKET
from drst_common.config import MODEL_DIR, RESULT_DIR
from drst_forecasting.models import SkWrapper, TorchWrapper
from drst_forecasting.dataset import build_sliding_window

def _load_selection():
    sel_key = f"{MODEL_DIR}/forecast/selected.json"
    raw = s3.get_object(Bucket=BUCKET, Key=sel_key)["Body"].read()
    sel = json.loads(raw.decode("utf-8"))
    kind = str(sel["selected_kind"]).lower()
    meta_key = sel["meta_key"]
    if kind in ("ridge", "xgboost"):
        wrapper = SkWrapper.load_from_s3(meta_key)
    else:
        wrapper = TorchWrapper.load_from_s3(meta_key)
    lookback = int(sel["lookback"]); horizon = int(sel["horizon"])
    return wrapper, lookback, horizon, sel

def _save_csv(key: str, df: pd.DataFrame):
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

def main():
    wrapper, L, H, sel = _load_selection()
    # 只取最后 1 个滑窗，multi_output=True 方便深度/传统统一
    X, _Y, _feats = build_sliding_window(L, H, take_last_n=1, multi_output=True)
    yhat = wrapper.predict(X)[0].tolist()
    yhat = yhat[:H]

    ts = int(time.time())
    # CSV
    df = pd.DataFrame({"step": list(range(1, len(yhat)+1)), "pred": np.asarray(yhat, dtype=np.float32)})
    _save_csv(f"{RESULT_DIR}/forecasting/forecast_{ts}.csv", df)
    # JSON
    save_bytes(
        f"{RESULT_DIR}/forecasting/forecast_{ts}.json",
        json.dumps({"horizon": len(yhat), "pred": yhat, "selected": sel}, ensure_ascii=False, indent=2).encode("utf-8"),
        "application/json",
    )
    # ready flag
    save_bytes(f"{RESULT_DIR}/forecast_ready.flag", b"1\n", "text/plain")
    print(f"[publish_forecast] wrote forecast_{ts}.csv/json and forecast_ready.flag", flush=True)

if __name__ == "__main__":
    main()
