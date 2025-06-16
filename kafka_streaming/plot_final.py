#!/usr/bin/env python3
"""
kafka_streaming/plot_final.py  – merge multi-consumer traces by timestamp (MinIO-only)
直接从 MinIO 列举并加载 *_inference_trace.npz
按时间戳合并并排序
保持 offline bridge + draw_three_phases() 接口
"""
import time
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shared.config       import RESULT_DIR
from shared.minio_helper import load_np, save_bytes, s3, BUCKET
from ml.plot_report      import draw_three_phases

# 1. 等离线 bridge artefacts
need_off = {"br_true": "bridge_true.npy", "br_pred": "bridge_pred.npy"}
offline = {}
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
print(f"[plot] bridge_ok  true={offline['br_true'].shape}  pred={offline['br_pred'].shape}")

# 2. 从 MinIO 收集所有 consumer trace
def _load_all_traces_minio() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{RESULT_DIR}/")
    traces = []
    for item in objs.get("Contents", []):
        key = item["Key"]
        if key.endswith("_inference_trace.npz"):
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            z = np.load(io.BytesIO(raw))
            traces.append((z["ts"], z["pred_adj"], z["pred_orig"], z["true"]))
    if not traces:
        return (np.array([]),) * 3
    # 合并并排序
    ts_all, adj_all, orig_all, true_all = [], [], [], []   # z["ts"], z["pred_adj"], z["pred_orig"], z["true"])
    for ts, adj, orig, tru in traces:
        ts_all.append(ts); adj_all.append(adj)
        orig_all.append(orig); true_all.append(tru)
    ts_all   = np.concatenate(ts_all)
    adj_all  = np.concatenate(adj_all)
    orig_all = np.concatenate(orig_all)
    true_all = np.concatenate(true_all)
    order = np.argsort(ts_all)   # 按时间戳排序
    return adj_all[order], orig_all[order], true_all[order]   # 按照adj_all 的顺序返回

# 阻塞等待直到拿到在线数据或超时
t0 = time.time()
while True:
    y_pred_adj, y_pred_orig, y_true = _load_all_traces_minio()
    if y_true.size:
        break
    if time.time() - t0 > CONSUMER_WAIT_S:
        print("[plot] timeout – 仍无 online data，后续图表仅离线部分")
        y_pred_adj = y_pred_orig = y_true = np.array([])
        break
    time.sleep(SLEEP_INTERVAL)

# 3. 合并 bridge + online
bridge_true = offline['br_true']
bridge_pred = offline['br_pred']

y_true_full      = np.concatenate([bridge_true,      y_true])
y_pred_orig_full = np.concatenate([bridge_pred,      y_pred_orig])
y_pred_adj_full  = np.concatenate([bridge_pred,      y_pred_adj])

# 4. overview_full.png（用 index 作横坐标）
x_idx = np.arange(len(y_true_full))

fig, ax = plt.subplots(facecolor="white", figsize=(14, 6))
ax.set_facecolor("white")

ax.plot(x_idx, y_pred_adj_full , "g-",  marker="o", ms=2, lw=1, label="Adjusted Prediction")
ax.plot(x_idx, y_true_full     , "b-",              lw=1,       label="Real data")
ax.plot(x_idx, y_pred_orig_full, "r--", marker="o", ms=2, lw=1, label="Original Prediction")

ax.set_xlabel("Sample Index", fontsize=14)
ax.set_ylabel("Throughput (Mbps)", fontsize=14)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
ax.grid(True, ls="--", lw=0.4)
ax.legend(loc="lower right")

plt.tight_layout()
png_full = "overview_full.png"
plt.savefig(png_full, dpi=150)
with open(png_full, "rb") as f:
    save_bytes(f"{RESULT_DIR}/{png_full}", f.read(), "image/png")
print("[plot] overview_full.png saved")

# 5. Phase-1/2/3 注解图（维持不变）
if y_true.size:
    png_phase = "report_phases.png"
    draw_three_phases(
        bridge_true   = bridge_true,
        bridge_pred   = bridge_pred,
        dag_true      = y_true,
        dag_pred_orig = y_pred_orig,
        dag_pred_adj  = y_pred_adj,
        dt            = 1.0,   # index 模式下 dt 不再使用，填任意值
        save_path     = png_phase
    )
    with open(png_phase, "rb") as f:
        save_bytes(f"{RESULT_DIR}/{png_phase}", f.read(), "image/png")
    print("[plot] report_phases.png saved")
else:
    print("[plot] skip Phase-3 graph – no online data")

# 6. KFP metadata 占位
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[plot] done.")