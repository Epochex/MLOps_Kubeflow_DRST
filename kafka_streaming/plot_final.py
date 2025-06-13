#!/usr/bin/env python3
"""
kafka_streaming/plot_final.py
────────────────────────────────────────────────────────────
• 从 PVC 抓离线 & 在线三个数组：
    - baseline = 原始 offline 模型
    - adjusted = 动态热重载模型
    - true     = 真实值
• 采样间隔 dt = env(MSG_SLEEP)（Producer 侧发送 sleep）
• 生成 3 张图（overview_full / report_detail / report_windowed）
  均使用 "秒" 为横坐标
• metrics CSV 合并逻辑保留
"""
import os, time, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shared.config       import RESULT_DIR, PLOT_WINDOW_BEFORE, PLOT_WINDOW_AFTER
from shared.minio_helper import load_np, save_bytes
from ml.plot_report      import generate_report, generate_report_full

# ---------- 采样间隔（秒） ----------
dt = float(os.getenv("MSG_SLEEP", "0.1"))      # Producer 默认 0.1 s

# ---------- 等待 offline Artefacts ----------
root_mnt = f"/mnt/pvc/{RESULT_DIR}"
os.makedirs(root_mnt, exist_ok=True)

timeout, t0 = 600, time.time()
REQUIRED_OFF = {
    "br_true": f"{RESULT_DIR}/bridge_true.npy",
    "br_pred": f"{RESULT_DIR}/bridge_pred.npy",
}
while True:
    try:
        offline = {k: load_np(path) for k, path in REQUIRED_OFF.items()}
        if all(arr.size > 0 for arr in offline.values()):
            break
    except Exception:
        pass
    if time.time() - t0 > timeout:
        print("[plot] timeout waiting for offline arrays – skip"); exit(0)
    time.sleep(5)
print("[plot] fetched offline arrays")

# ---------- 收集所有 inference 副本 ----------
pods = [
    d for d in os.listdir(root_mnt)
    if os.path.isdir(os.path.join(root_mnt, d)) and d not in ("timing",)
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
full_png = os.path.join(root_mnt, "overview_full.png")
generate_report_full(
    bridge_true       = offline["br_true"],
    bridge_pred_orig  = offline["br_pred"],
    dag_pred_orig     = y_pred_orig,
    dag_pred_new      = y_pred_adj,
    yd1               = y_true,
    dt                = dt,
    save_path         = full_png,
)
with open(full_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/overview_full.png", fp.read(), "image/png")

# ---------- 2) 过滤后细节图 ----------
detail_png = os.path.join(root_mnt, "report_detail.png")
generate_report(
    bridge_true       = offline["br_true"],
    bridge_pred_orig  = offline["br_pred"],
    dag1_pred_orig    = y_pred_orig,
    y_pred_dag1_new   = y_pred_adj,
    yd1               = y_true,
    dt                = dt,
    save_path         = detail_png,
)
with open(detail_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/report_final.png", fp.read(), "image/png")

# ---------- 3) 窗口图：围绕 correction_index ± window ----------
bt = offline["br_true"];  bp = offline["br_pred"]
y_true_full      = np.concatenate([bt, y_true])
y_pred_orig_full = np.concatenate([bp, y_pred_orig])
y_pred_adj_full  = np.concatenate([bp, y_pred_adj])

full_len   = len(y_true_full)
bridge_len = len(bt)
dag1_len   = len(y_true)

thr_drift = 0.46
if dag1_len > 0:
    err_ratio = np.abs(y_true - y_pred_orig) / np.maximum(y_true, 1e-8)
    bad_idx = dag1_len if err_ratio.max() <= thr_drift else int(np.argmax(err_ratio > thr_drift))
    drift_index = bridge_len + bad_idx
else:
    drift_index = bridge_len
correction_offset = 38
correction_index = min(full_len, drift_index + correction_offset)

# 时间坐标
times = np.arange(full_len) * dt
start = max(0, correction_index - PLOT_WINDOW_BEFORE)
end   = min(full_len, correction_index + PLOT_WINDOW_AFTER)

fig, ax = plt.subplots(facecolor="white", figsize=(12, 6))
ax.set_facecolor("white")
ax.plot(times[start:end], y_pred_adj_full[start:end],
        "g-", marker="o", markersize=3, linewidth=1.0, label="Adjusted Prediction")
ax.plot(times[start:end], y_true_full[start:end],
        "b-", linewidth=1.0, label="Real data")
ax.plot(times[start:end], y_pred_orig_full[start:end],
        "r--", marker="o", markersize=3, linewidth=1.0, label="Original Prediction")

ax.axvline(times[correction_index], color="magenta", ls="--", lw=2, label="Correction Point")
ax.text(times[correction_index], ax.get_ylim()[1]*0.95, f"{times[correction_index]:.1f}s",
        ha="center", va="top", color="magenta")

ax.set_xlabel("Time (s)", fontsize=14)
ax.set_ylabel("Throughput (Mbps)", fontsize=14)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
ax.grid(True, linestyle="--", linewidth=0.4)
ax.legend(loc="lower right", fontsize=12)
plt.tight_layout()

window_png = os.path.join(root_mnt, "report_windowed.png")
plt.savefig(window_png, dpi=150); plt.close(fig)
with open(window_png, "rb") as fp:
    save_bytes(f"{RESULT_DIR}/report_windowed.png", fp.read(), "image/png")

print("[plot] uploaded overview, detail and windowed reports to MinIO")

# ---------- 4) 合并 inference metrics CSV – 保留原逻辑 ----------
all_paths = glob.glob(os.path.join(root_mnt, "*_infer_metrics.csv"))
if all_paths:
    df_all = pd.concat(
        (pd.read_csv(p).assign(pod_name=os.path.basename(p).split("_infer")[0])
         for p in all_paths), ignore_index=True)
    df_all["utc"] = pd.to_datetime(df_all["utc"])
    df_all.sort_values("utc", inplace=True)
    merged = os.path.join(root_mnt, "all_infer_metrics.csv")
    df_all.to_csv(merged, index=False)
    with open(merged, "rb") as fp:
        save_bytes(f"{RESULT_DIR}/all_infer_metrics.csv", fp.read(), "text/csv")
    print("[plot] merged inference metrics → all_infer_metrics.csv")

# ---------- 5) 写 KFP V2 metadata.json ----------
meta_dir = "/tmp/kfp_outputs"
os.makedirs(meta_dir, exist_ok=True)
with open(f"{meta_dir}/output_metadata.json", "w") as f:
    json.dump({}, f)
print("[plot] all tasks complete.")
