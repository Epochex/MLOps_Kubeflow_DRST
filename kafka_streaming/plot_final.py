#!/usr/bin/env python3
# kafka_streaming/plot_final.py
# ------------------------------------------------------------
# 汇总离线 + 在线两段预测，生成：
#   • report_final.png   – 截取 200 点的细节三阶段对比（图 3）
#   • overview_full.png  – 全时序总览
# ------------------------------------------------------------
import os, time, numpy as np
from shared.config       import RESULT_DIR
from shared.minio_helper import load_np, save_bytes
from ml.plot_report      import generate_report, generate_report_full

# ---------- 需要等待的文件 ----------
root_mnt = f"/mnt/pvc/{RESULT_DIR}"
REQUIRED = {
    "br_true"       : f"{RESULT_DIR}/bridge_true.npy",
    "br_pred"       : f"{RESULT_DIR}/bridge_pred.npy",
    "dg_pred_orig"  : f"{RESULT_DIR}/inference_pred_orig.npy",
    "dg_pred_adj"   : f"{RESULT_DIR}/inference_pred_adj.npy",
    "dg_true"       : f"{RESULT_DIR}/inference_true.npy",
}
timeout, t0 = 600, time.time()
while True:
    try:
        arrays = {k: load_np(v) for k, v in REQUIRED.items()}
        if all(a.size > 0 for a in arrays.values()):
            break
    except Exception:
        pass
    if time.time() - t0 > timeout:
        print("[plot] timeout waiting for arrays – skip"); exit(0)
    time.sleep(5)
print("[plot] fetched arrays from MinIO")

# ---------- 细节图（200 点三条线 + 三阶段垂线） ----------
os.makedirs(root_mnt, exist_ok=True)
detail_png = f"{root_mnt}/report_final.png"
generate_report(
    bridge_true        = arrays["br_true"],
    bridge_pred_orig   = arrays["br_pred"],
    dag1_pred_orig     = arrays["dg_pred_orig"],
    y_pred_dag1_new    = arrays["dg_pred_adj"],      # ✅ 真正的“绿色”曲线
    yd1                = arrays["dg_true"],
    save_path          = detail_png,
)
with open(detail_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/report_final.png", fp.read(), "image/png")

# ---------- 总览图（全时序三条线，不画垂线） ----------
full_png = f"{root_mnt}/overview_full.png"
generate_report_full(
    bridge_true          = arrays["br_true"],
    bridge_pred_orig     = arrays["br_pred"],
    dag_pred_orig        = arrays["dg_pred_orig"],
    dag_pred_new         = arrays["dg_pred_adj"],
    yd1                  = arrays["dg_true"],
    save_path            = full_png,
)
with open(full_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/overview_full.png", fp.read(), "image/png")

print("[plot] uploaded both figures to MinIO")

import json, os
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json","w") as f:
    json.dump({}, f)