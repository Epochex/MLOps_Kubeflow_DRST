#!/usr/bin/env python3
"""
kafka_streaming/plot_final.py
────────────────────────────────────────────────────────────
• 合并所有副本／offline 输出
• 生成三张图：
    - report_full.png  （全时序总览）
    - report_detail.png（原有过滤后 200 点）
    - report_windowed.png（围绕校正完成点 ± window）
"""
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shared.config       import RESULT_DIR, PLOT_WINDOW_BEFORE, PLOT_WINDOW_AFTER
from shared.minio_helper import load_np, save_bytes
from ml.plot_report      import generate_report, generate_report_full

# ---------- 等待文件准备 & 确保目录存在 ----------
root_mnt = f"/mnt/pvc/{RESULT_DIR}"
# 新增：即使还没有任何文件，也要创建这个目录
os.makedirs(root_mnt, exist_ok=True)

timeout, t0 = 600, time.time()

REQUIRED_OFF = {
    "br_true": f"{RESULT_DIR}/bridge_true.npy",
    "br_pred": f"{RESULT_DIR}/bridge_pred.npy",
}
while True:
    try:
        offline = {k: load_np(v) for k, v in REQUIRED_OFF.items()}
        if all(arr.size > 0 for arr in offline.values()):
            break
    except:
        pass
    if time.time() - t0 > timeout:
        print("[plot] timeout waiting for offline arrays – skip"); exit(0)
    time.sleep(5)
print("[plot] fetched offline arrays")

# ---------- 扫描所有 inference 副本 ----------
pods = [
    d for d in os.listdir(root_mnt)
    if os.path.isdir(os.path.join(root_mnt, d))
       and d not in ("timing",)
]
print(f"[plot] found inference pods: {pods}")

pred_adj_list, pred_orig_list, true_list = [], [], []
for pod in sorted(pods):
    base = os.path.join(root_mnt, pod)
    try:
        pred_adj_list.append(np.load(f"{base}/inference_pred_adj.npy"))
        pred_orig_list.append(np.load(f"{base}/inference_pred_orig.npy"))
        true_list.append(np.load(f"{base}/inference_true.npy"))
    except FileNotFoundError:
        print(f"[plot] missing files in {pod}, skipping")

y_pred_adj  = np.concatenate(pred_adj_list)  if pred_adj_list  else np.array([])
y_pred_orig = np.concatenate(pred_orig_list) if pred_orig_list else np.array([])
y_true      = np.concatenate(true_list)      if true_list      else np.array([])

# ---------- 1) 全量总览图 ----------
os.makedirs(root_mnt, exist_ok=True)
full_png = f"{root_mnt}/overview_full.png"
generate_report_full(
    bridge_true       = offline["br_true"],
    bridge_pred_orig  = offline["br_pred"],
    dag_pred_orig     = y_pred_orig,
    dag_pred_new      = y_pred_adj,
    yd1               = y_true,
    save_path         = full_png,
)
with open(full_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/overview_full.png", fp.read(), "image/png")

# ---------- 2) 原有过滤后细节（200 点） ----------
detail_png = f"{root_mnt}/report_detail.png"
generate_report(
    bridge_true       = offline["br_true"],
    bridge_pred_orig  = offline["br_pred"],
    dag1_pred_orig    = y_pred_orig,
    y_pred_dag1_new   = y_pred_adj,
    yd1               = y_true,
    save_path         = detail_png,
)
with open(detail_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/report_final.png", fp.read(), "image/png")

# ---------- 3) 窗口图：围绕校正完成点 ± PLOT_WINDOW_* ----------
# 重新拼 full-arrays
bt = offline["br_true"]
bp = offline["br_pred"]
y_true_full      = np.concatenate([bt, y_true])
y_pred_orig_full = np.concatenate([bp, y_pred_orig])
y_pred_adj_full  = np.concatenate([bp, y_pred_adj])
full_len = len(y_true_full)
bridge_len = len(bt)
dag1_len   = len(y_true)

# 计算 drift_index 与 correction_index（同 generate_report 逻辑）
thr_drift = 0.15
if dag1_len > 0:
    err_ratio = np.abs(y_true - y_pred_orig) / np.maximum(y_true, 1e-8)
    if err_ratio.max() <= thr_drift:
        bad_idx = dag1_len
    else:
        bad_idx = int(np.argmax(err_ratio > thr_drift))
    drift_index     = bridge_len + bad_idx
else:
    drift_index = bridge_len
correction_offset = 38
correction_index = min(full_len, drift_index + correction_offset)

# 定义窗口范围
start = max(0, correction_index - PLOT_WINDOW_BEFORE)
end   = min(full_len, correction_index + PLOT_WINDOW_AFTER)
x = np.arange(start, end)

fig, ax = plt.subplots(facecolor="white", figsize=(12, 6))
ax.set_facecolor("white")
ax.plot(x, y_pred_adj_full[start:end],
        "g-", marker="o", markersize=3, linewidth=1.0,
        label="Adjusted Prediction")
ax.plot(x, y_true_full[start:end],
        "b-", linewidth=1.0,
        label="Real data")
ax.plot(x, y_pred_orig_full[start:end],
        "r--", marker="o", markersize=3, linewidth=1.0,
        label="Original Prediction")
# 标记校正完成点
ax.axvline(correction_index, color="magenta", ls="--", lw=2, label="Correction Point")
ax.text(correction_index, ax.get_ylim()[1]*0.95,
        f"{correction_index}", ha="center", va="top", color="magenta")
ax.set_xlabel("Time series index", fontsize=14)
ax.set_ylabel("Throughput (Mbps)", fontsize=14)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
ax.grid(True, linestyle="--", linewidth=0.4)
ax.legend(loc="lower right", fontsize=12)
plt.tight_layout()

windowed_png = f"{root_mnt}/report_windowed.png"
plt.savefig(windowed_png, dpi=150)
plt.close(fig)

with open(windowed_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/report_windowed.png", fp.read(), "image/png")

print("[plot] uploaded overview, detail and windowed reports to MinIO")

# 写 KFP V2 metadata
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json","w") as f:
    json.dump({}, f)
