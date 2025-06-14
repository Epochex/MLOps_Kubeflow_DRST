#!/usr/bin/env python3
"""
kafka_streaming/plot_final.py  – merge multi-consumer traces by timestamp
────────────────────────────────────────────────────────────────────────────
• 递归扫描 PVC，如无则从 MinIO 兜底，读取 *_inference_trace.npz
• 按时间戳合并多个 consumer 的在线数据并排序
• 仍保持离线 bridge + draw_three_phases() 的绘图接口
"""
import os
import time
import glob
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shared.config        import RESULT_DIR
from shared.minio_helper  import load_np, save_bytes, s3, BUCKET
from ml.plot_report       import draw_three_phases

# ─── 全局参数 ───────────────────────────────────────────
dt        = float(os.getenv("MSG_SLEEP", "0.1"))
root_pvc  = f"/mnt/pvc/{RESULT_DIR}"
os.makedirs(root_pvc, exist_ok=True)

# 1. 等离线 bridge artefacts ------------------------------------------------
need_off = {"br_true": "bridge_true.npy",
            "br_pred": "bridge_pred.npy"}
offline  = {}
t0 = time.time()
while True:
    try:
        offline = {k: load_np(f"{RESULT_DIR}/{v}") for k, v in need_off.items()}
        if all(arr.size for arr in offline.values()):
            break
    except Exception:
        pass
    if time.time() - t0 > 600:
        raise RuntimeError("[plot] 10 min 仍未等到 bridge artefacts，终止")
    time.sleep(3)

print(f"[plot] bridge_ok  true={offline['br_true'].shape}  "
      f"pred={offline['br_pred'].shape}")

# 2. 收集 consumer trace（PVC → MinIO 兜底）-------------------------------
CONSUMER_WAIT_S = 600   # 最多等 10 min
SCAN_INTV_S     = 3

def _scan_trace_pvc() -> list[str]:
    """返回 PVC 内所有 *_inference_trace.npz 路径列表"""
    patt = os.path.join(root_pvc, "**", "*_inference_trace.npz")
    return glob.glob(patt, recursive=True)

def _scan_trace_minio() -> dict[str, bytes]:
    """返回 {basename: raw_bytes}"""
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{RESULT_DIR}/")
    out = {}
    for item in objs.get("Contents", []):
        key = item["Key"]
        if key.endswith("_inference_trace.npz"):
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            out[os.path.basename(key)] = raw
    return out

def _load_all_traces() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    合并所有 consumer 的 (adj, orig, true)，按时间戳排序
    若无任何 trace，返回三个空数组
    """
    traces = []

    # —— PVC ——  
    pvc_files = _scan_trace_pvc()
    for fp in pvc_files:
        z = np.load(fp)
        traces.append((z["ts"], z["pred_adj"], z["pred_orig"], z["true"]))

    # —— MinIO 兜底 ——  
    if not traces:
        raw_map = _scan_trace_minio()
        for raw in raw_map.values():
            z = np.load(io.BytesIO(raw))
            traces.append((z["ts"], z["pred_adj"], z["pred_orig"], z["true"]))

    if not traces:           # 仍然没数据
        return (np.array([]),) * 3

    # —— 合并并全局排序 ——  
    ts_all, adj_all, orig_all, true_all = [], [], [], []
    for ts, adj, orig, tru in traces:
        ts_all   .append(ts)
        adj_all  .append(adj)
        orig_all .append(orig)
        true_all .append(tru)

    ts_all   = np.concatenate(ts_all)
    adj_all  = np.concatenate(adj_all)
    orig_all = np.concatenate(orig_all)
    true_all = np.concatenate(true_all)

    order = np.argsort(ts_all)          # 升序
    return adj_all[order], orig_all[order], true_all[order]

# —— 阻塞等待直到拿到在线数据或超时 ——  
t0 = time.time()
while True:
    y_pred_adj, y_pred_orig, y_true = _load_all_traces()
    if y_true.size:      # 至少一个样本
        break
    if time.time() - t0 > CONSUMER_WAIT_S:
        print("[plot] timeout – 仍无 online data，后续图表仅离线部分")
        y_pred_adj = y_pred_orig = y_true = np.array([])
        break
    time.sleep(SCAN_INTV_S)

# 3. 合并 bridge + online --------------------------------------------------
bridge_true = offline["br_true"]
bridge_pred = offline["br_pred"]

y_true_full      = np.concatenate([bridge_true,      y_true])
y_pred_orig_full = np.concatenate([bridge_pred,      y_pred_orig])
y_pred_adj_full  = np.concatenate([bridge_pred,      y_pred_adj])
times = np.arange(len(y_true_full)) * dt

# 4. overview_full.png -----------------------------------------------------
fig, ax = plt.subplots(facecolor="white", figsize=(14, 6))
ax.set_facecolor("white")
ax.plot(times, y_pred_adj_full , "g-",  marker="o", ms=2, lw=1, label="Adjusted Prediction")
ax.plot(times, y_true_full     , "b-",              lw=1,       label="Real data")
ax.plot(times, y_pred_orig_full, "r--", marker="o", ms=2, lw=1, label="Original Prediction")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Throughput (Mbps)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y,_: f"{int(y)}"))
ax.grid(True, ls="--", lw=0.4)
ax.legend(loc="lower right")
plt.tight_layout()
png_full = os.path.join(root_pvc, "overview_full.png")
plt.savefig(png_full, dpi=150)
with open(png_full, "rb") as f:
    save_bytes(f"{RESULT_DIR}/overview_full.png", f.read(), "image/png")
print("[plot] overview_full.png saved")

# 5. Phase-1/2/3 注解图（若有 online）-------------------------------------
if y_true.size:
    png_phase = os.path.join(root_pvc, "report_phases.png")
    draw_three_phases(
        bridge_true   = bridge_true,
        bridge_pred   = bridge_pred,
        dag_true      = y_true,
        dag_pred_orig = y_pred_orig,
        dag_pred_adj  = y_pred_adj,
        dt            = dt,
        save_path     = png_phase
    )
    with open(png_phase, "rb") as f:
        save_bytes(f"{RESULT_DIR}/report_phases.png", f.read(), "image/png")
    print("[plot] report_phases.png saved")
else:
    print("[plot] skip Phase-3 graph – no online data")

# 6. KFP metadata 占位 -----------------------------------------------------
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[plot] done.")
