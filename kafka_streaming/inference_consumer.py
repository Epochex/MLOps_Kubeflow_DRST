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

from botocore.exceptions import ClientError     

# ---------- 常量 & 本地路径 ---------------------------------------------
MODEL_IMPROVE_EPS = float(os.getenv("MODEL_IMPROVE_EPS", "1.0"))  # %

TMP_DIR  = "/tmp/infer"                     # ← 改用 /tmp
os.makedirs(TMP_DIR, exist_ok=True)

pod_name = os.getenv("HOSTNAME", "infer")   # k8s 容器名

local_out = os.path.join(TMP_DIR, pod_name) # 每个 consumer 独立目录
os.makedirs(local_out, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
proc   = psutil.Process()

pred_orig_hist: List[float] = []
pred_adj_hist : List[float] = []
true_hist     : List[float] = []
ts_hist       : List[str]   = []

# ---------- scaler / PCA --------------------------------------------------
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")

try:
    with open(f"{local_out}/pca.pkl", "wb") as f:
        f.write(_fetch("pca.pkl"))
    pca = joblib.load(f"{local_out}/pca.pkl")
    use_pca = True
except Exception:
    pca, use_pca = None, False

# ---------- 完整模型加载工具函数 ------------------------------------------
def _load_full_model(key: str) -> tuple[nn.Module, bytes]:
    """
    从 MinIO 拉取 key 对应文件，确保它是完整 torch.save(model) 导出的 nn.Module。
    如果收到 OrderedDict（state_dict），_bytes_to_model 会抛 TypeError。
    返回 (model, raw_bytes)。
    """
    raw = _fetch(key)
    model = _bytes_to_model(raw).to(device)  # _bytes_to_model 已经 .eval()
    return model, raw

# ---------- baseline & adaptive 模型装载 ----------------------------------
baseline_model, base_raw = _load_full_model("baseline_model.pt")
baseline_in_dim          = baseline_model.net[0].in_features

current_model, curr_raw  = _load_full_model("model.pt")

current_model._val_acc15 = 0.0          # 初始基线

print(f"[infer:{pod_name}] baseline in_features = {baseline_in_dim}")
print(f"[infer:{pod_name}] adaptive model       = {current_model}")


model_sig        = hashlib.md5(curr_raw).hexdigest()
model_loading_ms = 0.0



# ---------- 热重载 --------------------------------------------------------
def _reload_model(force: bool = False):
    """拉取 latest.txt 指向的新模型；不存在就安静返回。"""
    global current_model, curr_raw, model_sig, model_loading_ms

    # 1) 没有 latest.txt 说明还没重训过 → 直接退出
    try:
        latest_raw = _fetch("latest.txt")        # models/latest.txt
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return                               # 首次启动的正常情况
        raise                                    # 其它异常继续抛

    # 2) 解析 latest.txt
    model_key, metrics_key = latest_raw.decode().strip().splitlines()

    # 3) 如果版本没变且不是强制刷新，就不加载
    raw = _fetch(model_key)
    sig = hashlib.md5(raw).hexdigest()
    if not force and sig == model_sig:
        return

    # 4) 读取验证集指标，只有当新模型 **明显更好** 才切换
    metrics = json.loads(_fetch(metrics_key).decode())
    new_acc = metrics.get("acc@0.15", 0.0)                      # ← 指标名统一
    old_acc = getattr(current_model, "_val_acc15", 0.0)
    if not force and (new_acc - old_acc) < MODEL_IMPROVE_EPS:
        print(f"[infer:{pod_name}] new acc={new_acc:.2f} ≤ old "
              f"{old_acc:.2f}+{MODEL_IMPROVE_EPS} → skip")
        return

    # 5) 真正热加载
    t0 = time.perf_counter()
    mdl = torch.load(io.BytesIO(raw), map_location=device).eval()
    current_model = mdl
    current_model._val_acc15 = new_acc              # 记录基准
    curr_raw, model_sig = raw, sig
    model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)

    log_metric(component="infer", event="hot_reload_runtime",
               model_loading_ms=model_loading_ms)
    print(f"[infer:{pod_name}] hot-reloaded → acc@0.15={new_acc:.2f}  "
          f"load={model_loading_ms} ms")



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

    # readiness flag：/tmp + MinIO
    flag_local = os.path.join(TMP_DIR, f"consumer_ready_{pod_name}.flag")
    open(flag_local, "w").close()
    print(f"[infer:{pod_name}] readiness flag →", flag_local)

    save_bytes(f"{RESULT_DIR}/consumer_ready_{pod_name}.flag",
               b"", "text/plain")

    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            producer_done.set()
            continue
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

def _align_to_dim(X_scaled: np.ndarray, in_dim: int) -> np.ndarray:
    """
    把 60-维 Scaler 特征裁剪/补零到目标 in_dim。
    """
    if in_dim == X_scaled.shape[1]:
        return X_scaled
    elif in_dim < X_scaled.shape[1]:
        return X_scaled[:, :in_dim]          # 截断
    else:
        pad = np.zeros((X_scaled.shape[0], in_dim - X_scaled.shape[1]),
                       dtype=np.float32)
        return np.concatenate([X_scaled, pad], axis=1)


def _make_input(model: nn.Module, X_scaled: np.ndarray) -> np.ndarray:
    """
    根据 **模型首层 in_features** 自动决定：
        • 用 PCA 特征       —— 若 in_dim == pca.n_components_
        • 用 Scaler 特征对齐 —— 否则
    这样无论 adaptive 模型是 6-维（PCA）还是 60-维（全特征）都能喂对输入。
    """
    in_dim = model.net[0].in_features
    if use_pca and in_dim == pca.n_components_:
        return pca.transform(X_scaled).astype(np.float32)
    return _align_to_dim(X_scaled, in_dim)

# ---------------------------  主循环  ------------------------------------
first_batch     = True
container_start = time.perf_counter()
IDLE_TIMEOUT_S  = 180
last_data_time  = time.time()
msg_total       = 0
correct_count   = 0
total_count     = 0

print(f"[infer:{pod_name}] consumer started…")

while True:
    batch = _take_batch()
    now   = time.time()

    if not batch:
        hot_reload()
        # 若已收到 producer_done 或长时间无数据则结束
        if producer_done.is_set() or (now - last_data_time) > IDLE_TIMEOUT_S:
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

    # 1) Extraction --------------------------------------------------------
    with Timer("Extraction", "infer"):
        rows_batch = list(batch)

    # 2) Preprocessing -----------------------------------------------------
    with Timer("Preprocessing", "infer"):
        X_raw = np.array(
            [[r["features"].get(c, 0.0) for c in FEATURE_COLS]
             for r in rows_batch],
            np.float32
        )
        X_scaled = scaler.transform(X_raw)

        # baseline 一直走 PCA → in_dim == baseline_in_dim
        X_base = pca.transform(X_scaled).astype(np.float32)       # (N, 6)

        # adaptive 根据当前模型 in_features 自适应选择 PCA or Scaler
        X_adpt = _make_input(current_model, X_scaled)

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

arr_adj   = np.asarray(pred_adj_hist , np.float32)
arr_orig  = np.asarray(pred_orig_hist, np.float32)
arr_true  = np.asarray(true_hist     , np.float32)
arr_ts    = np.asarray([
    datetime.fromisoformat(t.rstrip("Z")).timestamp()
    for t in ts_hist
], np.float64)

npz_local = os.path.join(local_out, "inference_trace.npz")
np.savez(npz_local,
         ts=arr_ts,
         pred_adj=arr_adj,
         pred_orig=arr_orig,
         true=arr_true)

with open(npz_local, "rb") as f:
    save_bytes(f"{RESULT_DIR}/{pod_name}_inference_trace.npz",
               f.read(), "application/octet-stream")
print(f"[infer:{pod_name}] trace npz saved – total {len(arr_true)} samples")

sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced, exiting.")

