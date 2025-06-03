#!/usr/bin/env python3
"""
kafka_streaming/plot_final.py
────────────────────────────────────────────────────────────
等待 inference_consumer 结束后 → 生成 report_final.png
同时把图片上传 MinIO
"""
import os, time, numpy as np
from shared.config  import RESULT_DIR
from shared.minio_helper import save_bytes
from ml.plot_report import generate_report

# 等待下游 .npy 写入
time.sleep(35)      # CONSUME_IDLE_S(10) + 缓冲

def _load(path):
    try: return np.load(path)
    except Exception: return np.empty((0,), np.float32)

root = f"/mnt/pvc/{RESULT_DIR}"
br_true = _load(f"{root}/bridge_true.npy")
br_pred = _load(f"{root}/bridge_pred.npy")
dg_pred = _load(f"{root}/inference_pred.npy")
dg_true = _load(f"{root}/inference_true.npy")
dg_adj  = _load(f"{root}/inference_pred_adjusted.npy")
if dg_adj.size == 0: dg_adj = dg_pred

if any(a.size == 0 for a in (br_true, br_pred, dg_pred, dg_true)):
    print("[plot] missing arrays – skip"); exit(0)

tmp_png  = "/tmp/report_tmp.png"
final_png= f"{root}/report_final.png"

generate_report(br_true, br_pred, dg_pred, dg_adj, dg_true, tmp_png)
os.replace(tmp_png, final_png)
print(f"[plot] saved → {final_png}")

# ───── 上传 MinIO ─────
with open(final_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/report_final.png", fp.read(), "image/png")
print("[plot] uploaded to MinIO")
