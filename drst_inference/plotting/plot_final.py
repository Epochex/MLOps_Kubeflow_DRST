#!/usr/bin/env python3
# drst_inference/plotting/plot_final.py
from __future__ import annotations
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, RESULT_DIR

def _latest_trace_key() -> Optional[str]:
    # Find the most recent *_inference_trace.npz under results/
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{RESULT_DIR}/")
    items = resp.get("Contents", [])
    cand = [o for o in items if o["Key"].endswith("_inference_trace.npz")]
    if not cand:
        return None
    cand.sort(key=lambda o: o["LastModified"], reverse=True)
    return cand[0]["Key"]

def main():
    key = _latest_trace_key()
    if not key:
        print("[plot_final] no inference trace npz found in results/")
        return
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    with io.BytesIO(obj["Body"].read()) as bio:
        npz = np.load(bio)
        ts = npz["ts"]
        pred_adj  = npz["pred_adj"]
        pred_orig = npz["pred_orig"]
        y_true    = npz["true"]

    # 1) Time-series comparison
    plt.figure(figsize=(10, 4.5))
    plt.plot(ts, y_true, label="truth")
    plt.plot(ts, pred_orig, label="baseline")
    plt.plot(ts, pred_adj, label="adaptive")
    plt.xlabel("timestamp (s, epoch)")
    plt.ylabel("value")
    plt.title("Inference trace")
    plt.legend(loc="best")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    save_bytes(f"{RESULT_DIR}/plot_final_timeseries.png", buf.read(), "image/png")

    # 2) Relative error histogram (adaptive)
    denom = np.maximum(np.abs(y_true), 1e-8)
    relerr = np.abs(pred_adj - y_true) / denom
    plt.figure(figsize=(6.5, 4.0))
    plt.hist(relerr, bins=50)
    plt.xlabel("relative error")
    plt.ylabel("count")
    plt.title("Adaptive relative error histogram")
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format="png", dpi=150)
    plt.close()
    buf2.seek(0)
    save_bytes(f"{RESULT_DIR}/plot_final_relerr.png", buf2.read(), "image/png")
    print("[plot_final] plots uploaded.")

if __name__ == "__main__":
    main()
