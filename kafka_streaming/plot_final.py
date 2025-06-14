#!/usr/bin/env python3
"""
kafka_streaming/plot_final.py  – 2025-06-14 debug rev
────────────────────────────────────────────────────────────
• 递归扫描 PVC + 若仍缺则从 MinIO 兜底
• 带详细 debug 打印，方便定位 “为什么没找到 online data”
"""
import os, time, glob, json, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

from shared.config       import RESULT_DIR
from shared.minio_helper import load_np, save_bytes, s3, BUCKET
from ml.plot_report      import draw_three_phases

# ─── 全局参数 ───────────────────────────────────────────
dt        = float(os.getenv("MSG_SLEEP", "0.1"))
root_pvc  = f"/mnt/pvc/{RESULT_DIR}"
os.makedirs(root_pvc, exist_ok=True)

 
# 1. 等离线 bridge artefacts
 
need_off  = {"br_true": "bridge_true.npy",
             "br_pred": "bridge_pred.npy"}
offline   = {}
t0        = time.time()
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

# 2. 收集 consumer 数组（PVC → MinIO 兜底）
 
CONSUMER_WAIT_S = 600          # 10 min
SCAN_INTV_S     = 3

def _scan_pvc() -> dict[str, dict[str,str]]:
    """
    返回 {prefix: {'true':path,'orig':path,'adj':path}}
    """
    patt = os.path.join(root_pvc, "**", "*_inference_true.npy")
    files_true = glob.glob(patt, recursive=True)
    mapping: dict[str, dict[str,str]] = defaultdict(dict)
    for f in files_true:
        prefix = os.path.basename(f).replace("_inference_true.npy", "")
        mapping[prefix]["true"] = f
    for kind in ("pred_orig", "pred_adj"):
        patt = os.path.join(root_pvc, "**", f"*_{kind}.npy")
        for f in glob.glob(patt, recursive=True):
            prefix = os.path.basename(f).replace(f"_{kind}.npy", "")
            mapping[prefix][kind[5:]] = f   # orig / adj
    return mapping

def _scan_minio() -> dict[str, dict[str,bytes]]:
    """
    若 PVC 没找到，再直接从 MinIO 把字节读出来放 BytesIO。
    """
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{RESULT_DIR}/")
    mapping: dict[str, dict[str,bytes]] = defaultdict(dict)
    for item in objs.get("Contents", []):
        key = item["Key"]
        if not key.endswith(".npy"):
            continue
        fname  = os.path.basename(key)
        if "_inference_" not in fname:
            continue
        if fname.endswith("_inference_true.npy"):
            prefix = fname.replace("_inference_true.npy","")
            kind   = "true"
        elif fname.endswith("_inference_pred_orig.npy"):
            prefix = fname.replace("_inference_pred_orig.npy","")
            kind   = "orig"
        elif fname.endswith("_inference_pred_adj.npy"):
            prefix = fname.replace("_inference_pred_adj.npy","")
            kind   = "adj"
        else:
            continue
        buf = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        mapping[prefix][kind] = buf
    return mapping

def _load_arrays() -> tuple[np.ndarray,np.ndarray,np.ndarray] | None:
    ready: dict[str, dict[str,str]] = _scan_pvc()
    print(f"[plot][debug] pvc scan: {len(ready)} prefixes")
    # 若 PVC 不足，再合并 MinIO 结果
    if not ready:
        ready_mio = _scan_minio()
        if ready_mio:
            print(f"[plot][debug] fallback minio prefixes={len(ready_mio)}")
        # 把字节流写到 BytesIO 直接 np.load
        for p, d in ready_mio.items():
            if all(k in d for k in ("true","orig","adj")):
                ready[p] = {
                    "true": io.BytesIO(d["true"]),
                    "orig": io.BytesIO(d["orig"]),
                    "adj" : io.BytesIO(d["adj"])
                }

    y_true, y_orig, y_adj = [], [], []
    missing = 0
    for p, d in ready.items():
        if not all(k in d for k in ("true","orig","adj")):
            missing += 1
            continue
        try:
            a_t = np.load(d["true"]) if isinstance(d["true"], io.BytesIO) else np.load(d["true"])
            a_o = np.load(d["orig"]) if isinstance(d["orig"], io.BytesIO) else np.load(d["orig"])
            a_a = np.load(d["adj" ]) if isinstance(d["adj" ], io.BytesIO) else np.load(d["adj" ])
        except Exception as e:
            print(f"[plot][warn] load {p} failed: {e}")
            continue
        if a_t.size:
            y_true.append(a_t); y_orig.append(a_o); y_adj.append(a_a)

    if y_true:
        print(f"[plot] collected {len(y_true)} consumers  "
              f"(PVC miss={missing})")
        return (np.concatenate(y_adj),
                np.concatenate(y_orig),
                np.concatenate(y_true))
    return None

t0 = time.time()
y_pred_adj = y_pred_orig = y_true = None
while True:
    res = _load_arrays()
    if res:
        y_pred_adj, y_pred_orig, y_true = res
        break
    if time.time() - t0 > CONSUMER_WAIT_S:
        print("[plot] timeout – 仍无 online data，后续图表仅离线部分")
        y_pred_adj = y_pred_orig = y_true = np.array([])
        break
    time.sleep(SCAN_INTV_S)

 
# 3. 合并 bridge + online
 
bridge_true = offline["br_true"];  bridge_pred = offline["br_pred"]

y_true_full      = np.concatenate([bridge_true,      y_true])
y_pred_orig_full = np.concatenate([bridge_pred,      y_pred_orig])
y_pred_adj_full  = np.concatenate([bridge_pred,      y_pred_adj])
times = np.arange(len(y_true_full)) * dt

 
# 4. overview_full.png
 
fig, ax = plt.subplots(facecolor="white", figsize=(14, 6))
ax.set_facecolor("white")
ax.plot(times, y_pred_adj_full , "g-",  marker="o", ms=2, lw=1, label="Adjusted Prediction")
ax.plot(times, y_true_full     , "b-",              lw=1,       label="Real data")
ax.plot(times, y_pred_orig_full, "r--", marker="o", ms=2, lw=1, label="Original Prediction")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Throughput (Mbps)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y,_: f"{int(y)}"))
ax.grid(True, ls="--", lw=0.4); ax.legend(loc="lower right")
plt.tight_layout()
png_full = os.path.join(root_pvc, "overview_full.png")
plt.savefig(png_full, dpi=150); plt.close(fig)
with open(png_full, "rb") as f:
    save_bytes(f"{RESULT_DIR}/overview_full.png", f.read(), "image/png")
print("[plot] overview_full.png saved")

 
# 5. Phase-1/2/3 注解图（若有 online）
 
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

 
# 6. KFP metadata 占位
 
os.makedirs("/tmp/kfp_outputs", exist_ok=True)
open("/tmp/kfp_outputs/output_metadata.json", "w").write("{}")
print("[plot] done.")
