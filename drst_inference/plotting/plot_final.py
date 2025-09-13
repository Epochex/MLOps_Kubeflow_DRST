#!/usr/bin/env python3
# drst_inference/plotting/plot_final.py
from __future__ import annotations
import os
import io
import numpy as np

# 显式使用 Agg 后端，避免容器内无 DISPLAY 报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, RESULT_DIR

def _list_trace_keys() -> List[str]:
    """列出 results/ 下所有 *_inference_trace.npz（来自 infer1/2/3）。"""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{RESULT_DIR}/")
    items = resp.get("Contents", []) or []
    return sorted([o["Key"] for o in items if o["Key"].endswith("_inference_trace.npz")])

def _load_npz_from_s3(key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    with io.BytesIO(obj["Body"].read()) as bio:
        npz = np.load(bio)
        ts = npz["ts"]
        pred_adj  = npz["pred_adj"]
        pred_orig = npz["pred_orig"]
        y_true    = npz["true"]
    return ts, pred_adj, pred_orig, y_true

def main():
    keys = _list_trace_keys()
    if not keys:
        print("[plot_final] no inference trace npz found in results/")
        return

    # 汇总所有副本的 trace
    all_ts, all_true, all_pred_orig, all_pred_adj = [], [], [], []
    for k in keys:
        try:
            ts, pa, po, yt = _load_npz_from_s3(k)
            all_ts.append(ts.astype(np.float64))
            all_true.append(yt.astype(np.float32))
            all_pred_orig.append(po.astype(np.float32))
            all_pred_adj.append(pa.astype(np.float32))
        except Exception as e:
            print(f"[plot_final] skip bad trace {k}: {e}")

    if not all_true:
        print("[plot_final] no valid traces")
        return

    ts = np.concatenate(all_ts)
    y_true    = np.concatenate(all_true)
    pred_orig = np.concatenate(all_pred_orig)
    pred_adj  = np.concatenate(all_pred_adj)

    # 1) Time-series comparison（合并后整体画一张）
    plt.figure(figsize=(10, 4.5))
    plt.plot(ts, y_true,    label="truth")
    plt.plot(ts, pred_orig, label="baseline")
    plt.plot(ts, pred_adj,  label="adaptive")
    plt.xlabel("timestamp (s, epoch)")
    plt.ylabel("value")
    plt.title("Inference trace (combined)")
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
    plt.title("Adaptive relative error histogram (combined)")
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format="png", dpi=150)
    plt.close()
    buf2.seek(0)
    save_bytes(f"{RESULT_DIR}/plot_final_relerr.png", buf2.read(), "image/png")
    print("[plot_final] plots uploaded.")

if __name__ == "__main__":
    main()
