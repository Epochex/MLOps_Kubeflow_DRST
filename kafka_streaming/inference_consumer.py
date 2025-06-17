#!/usr/bin/env python3
"""
kafka_streaming/inference_consumer.py
────────────────────────────────────────────────────────────
baseline  : offline 训练得到的 MLP Scaler→PCA→N 维输入
adaptive  : 热更新模型 Scaler 后直接吃 full_dim 特征
两路预测 + 真实值 全量记录，供 plot_final.py 直接拼 Phase-1/2/3
所有原有指标埋点 ,日志完全保留
"""

import os, io, json, time, queue, threading, hashlib
from datetime import datetime
from collections import deque
from typing import List


from time import sleep
import threading
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
model_lock = threading.Lock()
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

# ---------- 选特征 & scaler 加载 ------------------------------------------
import json, io, joblib

# 1) 拉取并加载与 output_rate 最相关的 top10 特征列表
raw_feats = _fetch("selected_feats.json")
SELECTED_FEATS = json.loads(raw_feats)
print(f"[infer:{pod_name}] using selected feats: {SELECTED_FEATS}")

# ① baseline_scaler：仅启动时加载一次，绝不在热重载中替换
baseline_scaler = joblib.load(io.BytesIO(_fetch("scaler.pkl")))

# ② adaptive_scaler：首次沿用 baseline，可在 hot-reload 时被替换
adaptive_scaler = baseline_scaler


# 2) 拉取并加载对应的 StandardScaler
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")


def _wait_for_latest_txt(timeout=120):
    print("[infer] polling for latest.txt …")
    for _ in range(timeout):
        try:
            s3.head_object(Bucket=BUCKET, Key=f"{MODEL_DIR}/latest.txt")
            print("[infer] ✅ latest.txt available")
            return
        except ClientError:
            sleep(1)
    raise TimeoutError("Timeout waiting for latest.txt")

_wait_for_latest_txt()

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
# ---------- 热重载 --------------------------------------------------------
GAIN_THR_PP = float(os.getenv("GAIN_THRESHOLD_PP", "0.5"))

def _reload_model(force: bool = False):
    global current_model, curr_raw, model_sig, model_loading_ms, adaptive_scaler

    print(f"[infer:{pod_name}] check update (force={force})")
    # 0) 先取 latest.txt
    try:
        latest_raw = _fetch("latest.txt")
    except ClientError as e:
        print(f"[infer:{pod_name}] no latest.txt yet: {e}")
        return

    parts = latest_raw.decode().strip().splitlines()
    if len(parts) < 2:
        print(f"[infer:{pod_name}] malformed latest.txt → {parts}")
        return
    model_key, metrics_key = parts[0], parts[1]
    scaler_key = parts[2] if len(parts) >= 3 else None
    print(f"[infer:{pod_name}] latest.txt → model={model_key}, metrics={metrics_key}, scaler={scaler_key}")

    # 1) 判断是不是同一个模型
    raw_model = _fetch(model_key)
    new_sig = hashlib.md5(raw_model).hexdigest()
    if not force and new_sig == model_sig:
        print(f"[infer:{pod_name}] signature unchanged ({new_sig}) → skip reload")
        return

    # 2) 看 accuracy 增益够不够
    metrics = json.loads(_fetch(metrics_key).decode())
    new_acc  = metrics.get("acc@0.15", 0.0)
    base_acc = metrics.get("baseline_acc@0.15", 0.0)
    gain_pp  = new_acc - base_acc
    print(f"[infer:{pod_name}] baseline_acc={base_acc:.2f}%, new_acc={new_acc:.2f}%, gain={gain_pp:.3f}pp (thr={GAIN_THR_PP}pp)")
    if not force and gain_pp < GAIN_THR_PP:
        print(f"[infer:{pod_name}] gain {gain_pp:.3f}pp < threshold → skip reload")
        return

    # 3) 真正加载新模型
    print(f"[infer:{pod_name}] loading new model from {model_key}")
    mdl = torch.load(io.BytesIO(raw_model), map_location=device).eval()
    new_scaler = None
    if scaler_key:
        try:
            new_scaler = joblib.load(io.BytesIO(_fetch(scaler_key)))
            print(f"[infer:{pod_name}] loaded new scaler {scaler_key}")
        except Exception as exc:
            print(f"[infer:{pod_name}] failed to load scaler {scaler_key}: {exc}")

    # 4) 原子更新
    with model_lock:
        current_model, curr_raw, model_sig = mdl, raw_model, new_sig
        if new_scaler is not None:
            adaptive_scaler = new_scaler

    model_loading_ms = 0.0
    print(f"[infer:{pod_name}] hot-reload complete → new signature {new_sig}")
    
    
def hot_reload():
    """异步触发热重载，不阻塞主推理循环。"""
    threading.Thread(target=_reload_model, daemon=True).start()



# ---------- Kafka 监听线程（带重试） -----------------------------------------------
import time
from kafka.errors import NoBrokersAvailable

q = queue.Queue()
producer_done = threading.Event()

def _create_consumer():
    """尝试重试连接 Kafka，多次失败后抛出 RuntimeError"""
    for attempt in range(1, 11):  # 最多重试 10 次
        try:
            return KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=",".join(KAFKA_SERVERS),
                group_id="cg-infer",
                auto_offset_reset=AUTO_OFFSET_RESET,
                enable_auto_commit=ENABLE_AUTO_COMMIT,
                value_deserializer=lambda m: json.loads(m.decode()),
                # 限制 api_version 自动探测时间，防止卡住
                api_version_auto_timeout_ms=10000,
            )
        except NoBrokersAvailable as e:
            print(f"[infer:{pod_name}] bootstrap brokers unavailable ({attempt}/10), retrying in 5s…")
            time.sleep(5)
    raise RuntimeError("[infer] Kafka still unreachable after 10 retries")

def _listener():
    # 1) 创建 consumer（内含重试）
    cons = _create_consumer()
    print(f"[infer:{pod_name}] KafkaConsumer created, beginning to poll…")

    # 2) readiness flag：/tmp + MinIO
    flag_local = os.path.join(TMP_DIR, f"consumer_ready_{pod_name}.flag")
    open(flag_local, "w").close()
    print(f"[infer:{pod_name}] readiness flag →", flag_local)
    save_bytes(f"{RESULT_DIR}/consumer_ready_{pod_name}.flag",
               b"", "text/plain")

    # 3) 持续接收消息
    for msg in cons:
        v = msg.value
        if v.get("producer_done"):
            producer_done.set()
            continue
        # 注入接收时间戳
        v["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        q.put(v)

# 启动监听线程（daemon 模式，不阻塞主线程）
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
    
    hot_reload()  # 让每个 consumer 等到真正收到哨兵后再结束。这样不会因为某些分区提前耗尽而“假死”或“早退
    batch = _take_batch()
    now   = time.time()
    
    if not batch:  #  只在 “收到 producer_done” 且 “本地队列已清空” 时才优雅收尾
        if producer_done.is_set() and q.empty():
            _reload_model(force=True)      # 最后再强制载一次最新模型
            break                          # graceful shutdown
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
            [[r["features"].get(c, 0.0) for c in SELECTED_FEATS]
            for r in rows_batch],
            dtype=np.float32
        )
        # baseline / adaptive 分别用自己的 scaler
        X_scaled_base = baseline_scaler.transform(X_raw)
        X_scaled_adpt = adaptive_scaler.transform(X_raw)

    # ── 线程安全地拿一份当前 adaptive 模型引用 ─────────────────────────────
    with model_lock:
        model_ref = current_model          # 只在临界区做引用拷贝
        
    # === 关键：让两路输入维度匹配 ============================
    X_base = _align_to_dim(X_scaled_base, baseline_in_dim) #  基线模型固定 10 维
    X_adpt = _make_input(model_ref,      X_scaled_adpt) # adaptive 可能是 10 或 60 维

        
        
    # 3) Inference ---------------------------------------------------------
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    with Timer("Inference_Engine", "infer"):
        with torch.no_grad():
            # Baseline 预测
            preds_base = baseline_model(
                torch.from_numpy(X_base).to(device)
            ).cpu().numpy().ravel()
            # Adaptive 预测（使用刚才线程安全取到的 model_ref）
            preds_adpt = model_ref(
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
np.savez(npz_local,     # 保存为 npz 格式 数据
         ts=arr_ts,     # 每条样本的发送时间戳
         pred_adj=arr_adj,  # 热更新模型的预测值序列
         pred_orig=arr_orig,  # 基线模型的预测值序列
         true=arr_true)  # 真实标签序列, 用来之后算吞吐量

with open(npz_local, "rb") as f:
    save_bytes(f"{RESULT_DIR}/{pod_name}_inference_trace.npz",
               f.read(), "application/octet-stream")
print(f"[infer:{pod_name}] trace npz saved – total {len(arr_true)} samples")

sync_all_metrics_to_minio()
print(f"[infer:{pod_name}] all metrics synced, exiting.")

