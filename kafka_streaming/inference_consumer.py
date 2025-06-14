#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
• baseline  : offline 训练得到的 MLP（Scaler→PCA→N 维输入）
• adaptive  : 热更新模型（Scaler 后直接吃 full_dim 特征）
• 两路预测 + 真实值全量落盘，供 plot_final.py 直接拼 Phase-1/2/3
• 所有原有指标埋点 / 日志完全保留
"""

import os, io, json, time, queue, threading, hashlib
from datetime import datetime
from collections import deque
from typing import List

import numpy as np
import joblib
import psutil
import torch
import torch.nn as nn
from kafka import KafkaConsumer

from shared.utils import _fetch, _bytes_to_model
from shared.metric_logger import log_metric, sync_all_metrics_to_minio
from shared.profiler      import Timer
from shared.minio_helper  import s3, save_bytes, save_np, BUCKET
from shared.config        import (
    KAFKA_TOPIC, KAFKA_SERVERS, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT
)
from shared.features      import FEATURE_COLS
from ml.model             import DynamicMLP, build_model


# ---------- 常量 ----------------------------------------------------------
done_flag = f"/mnt/pvc/{RESULT_DIR}/producer_done.flag"
pod_name  = os.getenv("HOSTNAME", "infer")
local_out = f"/mnt/pvc/{RESULT_DIR}/{pod_name}"
os.makedirs(local_out, exist_ok=True)

correct_count = 0
total_count   = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

pred_orig_hist: List[float] = []
pred_adj_hist : List[float] = []
true_hist     : List[float] = []
ts_hist       : List[str]   = []     


# ---------- scaler / PCA / baseline --------------------------------------
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

try:
    with open(f"{local_out}/pca.pkl", "wb") as f:
        f.write(_fetch("pca.pkl"))
    pca = joblib.load(f"{local_out}/pca.pkl"); use_pca = True
except Exception:
    pca, use_pca = None, False

# -- baseline & adaptive 都是“完整模型对象” --
baseline_model = torch.load(io.BytesIO(_fetch("baseline_model.pt")),
                            map_location=device).eval()
baseline_in_dim = baseline_model.net[0].in_features

curr_raw      = _fetch("model.pt")
current_model = torch.load(io.BytesIO(curr_raw),
                           map_location=device).eval()

print(f"[infer:{pod_name}] baseline in_features = {baseline_in_dim}")
print(f"[infer:{pod_name}] adaptive model       = {current_model}")

# —— 若 adaptive 权重“几乎空”，回退 baseline ——
w_mean = current_model.net[0].weight.abs().mean().item()
if w_mean < 1e-4:        # 阈值按需调整
    print(f"[infer:{pod_name}] Detected un-trained adaptive "
          f"( |w| mean={w_mean:.1e} ) → fallback to baseline")
    current_model = baseline_model

model_sig        = hashlib.md5(curr_raw).hexdigest()
model_loading_ms = 0.0


# ---------- 热重载 --------------------------------------------------------
def _reload_model(force: bool = False):
    global current_model, curr_raw, model_sig, model_loading_ms

    raw = _fetch("model.pt"); sig = hashlib.md5(raw).hexdigest()
    if not force and sig == model_sig:
        return

    t0 = time.perf_counter()
    mdl = torch.load(io.BytesIO(raw), map_location=device).eval()

    # 健康检查
    if mdl.net[0].weight.abs().mean().item() < 1e-4:
        print(f"[infer:{pod_name}] hot-reload found un-trained model → ignore")
        return                      # 不更新

    current_model = mdl
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)
    curr_raw, model_sig = raw, sig
    log_metric(component="infer", event="hot_reload_runtime",
               model_loading_ms=model_loading_ms)
    print(f"[infer:{pod_name}] hot-reloaded OK, load={model_loading_ms} ms")



def hot_reload():
    """异步触发热重载，不阻塞主推理循环。"""
    threading.Thread(target=_reload_model, daemon=True).start()


# ---------- Kafka 监听线程 -----------------------------------------------
q = queue.Queue()
producer_done = threading.Event()

def _listener():
    cons = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=",".join(KAFKA_SERVERS),
        group_id="cg-infer",
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=ENABLE_AUTO_COMMIT,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    flag = f"/mnt/pvc/{RESULT_DIR}/consumer_ready_{pod_name}.flag"
    open(flag, "w").close()
    print(f"[infer:{pod_name}] readiness flag → {flag}")

    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            producer_done.set(); continue
        v["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        q.put(v)

threading.Thread(target=_listener, daemon=True).start()

def _take_batch():
    buf = []
    try: buf.append(q.get(timeout=CONSUME_IDLE_S))
    except queue.Empty: return buf
    while len(buf) < BATCH_SIZE:
        try: buf.append(q.get_nowait())
        except queue.Empty: break
    return buf

# ---------- Forecasting (demo) -------------------------------------------
forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist: continue
        with Timer("Forecasting_Engine", "infer"): _ = float(np.mean(forecast_hist))
        log_metric(component="infer", event="forecasting_runtime")
threading.Thread(target=_forecast_loop, daemon=True).start()

def _align_adaptive_input(X_scaled: np.ndarray, model: nn.Module) -> np.ndarray:
    in_dim = model.net[0].in_features
    if in_dim == X_scaled.shape[1]:
        X_aligned = X_scaled
    elif in_dim < X_scaled.shape[1]:
        X_aligned = X_scaled[:, :in_dim]          # 截断
    else:
        pad = np.zeros((X_scaled.shape[0], in_dim - X_scaled.shape[1]),
                       dtype=np.float32)
        X_aligned = np.concatenate([X_scaled, pad], axis=1)

    # === DEBUG 2：输入是不是全 0？ =============================
    if not np.any(X_aligned):
        print(f"[infer:{pod_name}] DEBUG ②  aligned input **ALL ZERO** !")
    return X_aligned


# ---------- 主循环 --------------------------------------------------------
first_batch     = True
container_start = time.perf_counter()
IDLE_TIMEOUT_S  = 180
last_data_time  = time.time()
msg_total       = 0

# —— 新增：累计正确样本数 & 总样本数 ——  
correct_count = 0
total_count   = 0

print(f"[infer:{pod_name}] consumer started…")

while True:
    batch = _take_batch()
    now   = time.time()

    if not batch:
        hot_reload()
        if os.path.exists(done_flag) or (now - last_data_time) > IDLE_TIMEOUT_S:
            _reload_model(force=True)
            break
        time.sleep(0.3)
        continue

    last_data_time = now
    if first_batch:
        cold_ms = (time.perf_counter() - container_start) * 1000
        log_metric(component="infer", event="cold_start",
                   cold_start_ms=round(cold_ms, 3))
        first_batch = False

    # 1) Extraction
    with Timer("Extraction", "infer"):
        rows_batch = list(batch)

    # 2) Preprocessing
    with Timer("Preprocessing", "infer"):
        X_raw = np.array(
            [[r["features"].get(c, 0.0) for c in FEATURE_COLS]
             for r in rows_batch],
            np.float32
        )
        X_scaled = scaler.transform(X_raw)
        X_base   = pca.transform(X_scaled) if use_pca else X_scaled[:, :baseline_in_dim]
        X_adpt   = _align_adaptive_input(X_scaled, current_model)

    # 3) Inference
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            preds_base = baseline_model(
                torch.from_numpy(X_base).to(device)
            ).cpu().numpy().ravel()
            preds_adpt = current_model(
                torch.from_numpy(X_adpt).to(device)
            ).cpu().numpy().ravel()
    cpu1, t1 = proc.cpu_times(), time.perf_counter()

    # 4) Accuracy@0.2 统计
    labels = np.array([r["label"] for r in rows_batch], np.float32)
    errs   = np.abs(preds_adpt - labels) / np.maximum(labels, 1e-8)
    batch_correct = int((errs <= 0.2).sum())
    batch_total   = len(labels)
    correct_count += batch_correct
    total_count   += batch_total
    cum_acc = correct_count / total_count
    print(f"[infer:{pod_name}] accuracy@0.2 → batch {batch_correct}/{batch_total}, "
          f"cumulative {correct_count}/{total_count} = {cum_acc:.3f}")
    log_metric(
        component="infer",
        event="cumulative_accuracy",
        threshold=0.2,
        batch_correct=batch_correct,
        batch_total=batch_total,
        cumulative_correct=correct_count,
        cumulative_total=total_count,
        cumulative_accuracy=round(cum_acc, 3)
    )
    
    # ---------- DEBUG：误差分布（基线 vs. 自适应） ----------
    # ① 相对误差
    err_base = np.abs(preds_base - labels) / np.maximum(labels, 1e-8)
    err_adpt = np.abs(preds_adpt - labels) / np.maximum(labels, 1e-8)

    # ② 关键分位数
    pct = [50, 80, 90, 95, 99]
    base_q = np.percentile(err_base, pct).round(3)
    adpt_q = np.percentile(err_adpt, pct).round(3)

    print(
        f"[infer:{pod_name}]  Δ(relative) | "
        f"BASE p50/p80/p90/p95/p99 = {base_q.tolist()} | "
        f"ADPT = {adpt_q.tolist()}"
    )

    log_metric(
        component="infer",
        event="err_dist",
        base_p50=float(base_q[0]), base_p80=float(base_q[1]),
        base_p90=float(base_q[2]), base_p95=float(base_q[3]),
        base_p99=float(base_q[4]),
        adpt_p50=float(adpt_q[0]), adpt_p80=float(adpt_q[1]),
        adpt_p90=float(adpt_q[2]), adpt_p95=float(adpt_q[3]),
        adpt_p99=float(adpt_q[4]),
    )
    
    # ---------- 收集指标 & 日志 ------------------------------------------
    ts_hist.extend([r["send_ts"] for r in rows_batch])
    # 5) 原有指标 & 日志
    pred_orig_hist.extend(preds_base.tolist())
    pred_adj_hist .extend(preds_adpt.tolist())
    true_hist     .extend(labels.tolist())

    # 计算 RTT、CPU%、TPS 等
    rtts = []
    for r in rows_batch:
        if "send_ts" not in r:
            continue
        try:
            st = datetime.fromisoformat(r["send_ts"].rstrip("Z"))
            rt = (datetime.fromisoformat(r["_recv_ts"].rstrip("Z")) - st
                  ).total_seconds() * 1000
            rtts.append(rt)
        except Exception:
            pass
    avg_rtt = round(np.mean(rtts), 3) if rtts else 0.0
    wall_ms = (t1 - t0) * 1000
    cpu_used = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) * 1000
    cpu_pct = round(cpu_used / wall_ms, 2) if wall_ms else 0.0
    tp_s    = round(len(rows_batch) / (wall_ms or 1e-3), 3)

    log_metric(
        component="infer", event="batch_metrics",
        batch_size=len(rows_batch),
        latency_ms=round(wall_ms, 3),
        throughput_s=tp_s,
        cpu_pct=cpu_pct,
        gpu_mem_pct=0.0,
        model_loading_ms=model_loading_ms,
        container_latency_ms=round(wall_ms, 3),
        rtt_ms=avg_rtt
    )
    msg_total += len(rows_batch)
    print(f"[infer:{pod_name}] batch_metrics: "
          f"msg_total={msg_total}, latency={wall_ms:.3f}ms, "
          f"avg_rtt={avg_rtt}ms, cpu_pct={cpu_pct}/s")

    # 6) Forecast & Hot-reload 保持不变
    forecast_hist.extend(preds_adpt)
    hot_reload()

# 7) 收尾：保存数组 & 上传
print(f"[infer:{pod_name}] TOTAL processed {msg_total} samples, exit")


# ---------- 收尾：保存完整 trace 并上传 ----------------------------------
print(f"[infer:{pod_name}] TOTAL processed {msg_total} samples, exit")

# ★★★ 新增：把 send_ts 转成 epoch 秒，与预测一起保存 ★★★
arr_adj   = np.asarray(pred_adj_hist , np.float32)
arr_orig  = np.asarray(pred_orig_hist, np.float32)
arr_true  = np.asarray(true_hist     , np.float32)
arr_ts    = np.asarray([
    datetime.fromisoformat(t.rstrip("Z")).timestamp()  # iso → epoch
    for t in ts_hist
], np.float64)

npz_local = f"{local_out}/inference_trace.npz"
np.savez(npz_local,
         ts=arr_ts,
         pred_adj=arr_adj,
         pred_orig=arr_orig,
         true=arr_true)

with open(npz_local, "rb") as f:
    save_bytes(f"{RESULT_DIR}/{pod_name}_inference_trace.npz",
               f.read(), "application/octet-stream")
print(f"[infer:{pod_name}] trace npz saved – total {len(arr_true)} samples")

# 同步全局 metrics（保持不变）
sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced, exiting.")
