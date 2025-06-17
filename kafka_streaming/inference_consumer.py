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

import threading
import numpy as np
import joblib
import psutil
import torch
import torch.nn as nn
from kafka import KafkaConsumer, KafkaProducer
from shared.config import CONTROL_TOPIC, KAFKA_TOPIC, KAFKA_SERVERS
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
 
pause_event = threading.Event()
# ---------- 常量 & 本地路径 ---------------------------------------------

RETRAIN_TOPIC = os.getenv("RETRAIN_TOPIC", KAFKA_TOPIC + "_infer_count")   # kafka topic total 100
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

# 2) 拉取并加载对应的 StandardScaler
with open(f"{local_out}/scaler.pkl", "wb") as f:
    f.write(_fetch("scaler.pkl"))
scaler = joblib.load(f"{local_out}/scaler.pkl")


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
GAIN_THR_PP = float(os.getenv("GAIN_THRESHOLD_PP", "0.01"))  # ≥0.x 个百分点就换

# ---------- 热重载 --------------------------------------------------------
def _reload_model(force: bool = False):
    """
    热加载：仅当新模型相对 baseline 精度提升 >= GAIN_THR_PP
    或者 force=True 时才替换 current_model。
    🔒 线程安全：更新过程持 model_lock。
    """
    global current_model, curr_raw, model_sig, model_loading_ms

    try:
        # 1) 尝试拉取 latest.txt
        latest_raw = _fetch("latest.txt")
        model_key, metrics_key = latest_raw.decode().strip().splitlines()

        # 2) 拉取模型字节流并计算签名
        raw = _fetch(model_key)
        sig = hashlib.md5(raw).hexdigest()

        # 3) 非强制且签名未变 → 跳过
        if not force and sig == model_sig:
            return

        # 4) 读取验证集准确率，计算增益
        metrics = json.loads(_fetch(metrics_key).decode())
        new_acc  = metrics.get("acc@0.15",    0.0)
        base_acc = metrics.get("baseline_acc@0.15", 0.0)
        gain_pp  = new_acc - base_acc

        # 5) 非强制且增益不足 → 跳过
        if not force and gain_pp < GAIN_THR_PP:
            print(f"[infer:{pod_name}] Δ{gain_pp:+.3f} pp < {GAIN_THR_PP} → skip reload")
            return

        # 6) 加载新模型并原子更新
        t0 = time.perf_counter()
        mdl = torch.load(io.BytesIO(raw), map_location=device).eval()
        with model_lock:
            current_model      = mdl
            current_model._val_acc15 = new_acc
            curr_raw, model_sig     = raw, sig
        model_loading_ms = round((time.perf_counter() - t0) * 1000, 3)

        log_metric(component="infer", event="hot_reload_runtime",
                   model_loading_ms=model_loading_ms)
        print(
            f"[infer:{pod_name}] hot-reloaded ✓  "
            f"baseline={base_acc:.2f}% → new={new_acc:.2f}%  "
            f"(Δ{gain_pp:+.3f} pp)  load={model_loading_ms} ms"
        )

    except ClientError as e:
        # latest.txt 不存在 → 跳过；其他 S3 错误都 log 后跳过
        code = e.response.get("Error", {}).get("Code", "")
        if code == "NoSuchKey":
            print(f"[infer:{pod_name}] reload: no latest.txt → skip")
        else:
            print(f"[infer:{pod_name}] reload ClientError → {e}")
        return

    except Exception as e:
        # 网络中断、连接被拒等一切异常都 log 后跳过
        print(f"[infer:{pod_name}] reload unexpected error → {e}")
        return



# ---------- Kafka 监听线程（带重试） -----------------------------------------------
import time
from kafka.errors import NoBrokersAvailable

q = queue.Queue()
producer_done = threading.Event()

def _create_consumer():
    for attempt in range(1, 11):
        try:
            cons = KafkaConsumer(
                KAFKA_TOPIC, CONTROL_TOPIC,
                bootstrap_servers=",".join(KAFKA_SERVERS),
                group_id="cg-infer",
                auto_offset_reset=AUTO_OFFSET_RESET,
                enable_auto_commit=ENABLE_AUTO_COMMIT,
                value_deserializer=lambda m: json.loads(m.decode()),
                api_version_auto_timeout_ms=10000,
            )
            return cons
        except NoBrokersAvailable:
            print(f"[infer:{pod_name}] bootstrap brokers unavailable ({attempt}/10), retrying in 5s…")
            time.sleep(5)
    raise RuntimeError("[infer] Kafka still unreachable after 10 retries")

def _listener():
    # 0) 等待 producer_ready.flag （防止抢跑）
    MAX_WAIT_FOR_PRODUCER = 600
    WAIT_STEP = 5
    flag_key = f"{RESULT_DIR}/producer_ready.flag"

    def _wait_for_producer_flag():
        for i in range(0, MAX_WAIT_FOR_PRODUCER, WAIT_STEP):
            try:
                s3.head_object(Bucket=BUCKET, Key=flag_key)
                print(f"[infer:{pod_name}] ✓ detected producer_ready.flag in MinIO")
                return
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    print(f"[infer:{pod_name}] MinIO error → {e}")
                    break
            time.sleep(WAIT_STEP)
        raise TimeoutError(f"[infer:{pod_name}] ❌ wait for producer_ready.flag timeout after {MAX_WAIT_FOR_PRODUCER}s")

    _wait_for_producer_flag()

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
        # —— 控制消息 —— 
        if "retrain" in v:
            if v["retrain"] == "start":
                print("[infer] PAUSE consumption for retrain")
                pause_event.set()
            elif v["retrain"] == "end":
                print("[infer] RESUME consumption after retrain")
                pause_event.clear()
            continue
        if v.get("producer_done"):
            producer_done.set()
            continue
        # 注入接收时间戳
        v["_recv_ts"] = datetime.utcnow().isoformat() + "Z"
        q.put(v)


# -----------------------------------------------------------------------------
# 周期性后台热重载（每 RELOAD_INTERVAL_S 秒调用一次 _reload_model）
# -----------------------------------------------------------------------------
RELOAD_INTERVAL_S = int(os.getenv("RELOAD_INTERVAL_S", "30"))

def _reload_daemon():
    while True:
        time.sleep(RELOAD_INTERVAL_S)
        try:
            _reload_model()
        except Exception as e:
            # 理论上 _reload_model 已经吞掉所有异常，这里做兜底
            print(f"[infer:{pod_name}] reload daemon error → {e}")

# 启动守护线程
threading.Thread(target=_reload_daemon, daemon=True).start()

# 废弃原 per-batch 热重载，将 hot_reload 改为空实现
def hot_reload():
    # 每个 batch 不再触发热重载，改用上面的守护线程
    pass
# -----------------------------------------------------------------------------

def _take_batch():
    buf = []
    try: buf.append(q.get(timeout=CONSUME_IDLE_S))
    except queue.Empty: return buf
    while len(buf) < BATCH_SIZE:
        try: buf.append(q.get_nowait())
        except queue.Empty: break
    return buf


trigger_prod = KafkaProducer(
    bootstrap_servers=",".join(KAFKA_SERVERS),
    value_serializer=lambda m: json.dumps(m).encode(),
)

# ---------- Forecasting (demo) -------------------------------------------
forecast_hist = deque(maxlen=300)
def _forecast_loop():
    while True:
        time.sleep(30)
        if not forecast_hist: continue
        with Timer("Forecasting_Engine", "infer"): _ = float(np.mean(forecast_hist))
        log_metric(component="infer", event="forecasting_runtime")
threading.Thread(target=_forecast_loop, daemon=True).start()


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
    # 如果正在 retrain，则阻塞在这里，直到收到 resume 信号
    while pause_event.is_set():
        time.sleep(1)
    batch = _take_batch()
    now   = time.time()
    
    if not batch:  #  只在 “收到 producer_done” 且 “本地队列已清空” 时才优雅收尾
        if producer_done.is_set() and q.empty():
            try:
                _reload_model(force=True)   # 最后再强制载一次最新模型
            except Exception as e:
                print(f"[infer:{pod_name}] final reload error → {e}")
            print(f"[infer:{pod_name}] graceful shutdown")
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
        # ① 提取 top-10 原始特征
        X_raw = np.array(
            [[r["features"].get(c, 0.0) for c in SELECTED_FEATS]
             for r in rows_batch],
            dtype=np.float32
        )
        # ② 标准化
        X_scaled = scaler.transform(X_raw)

    # ── 线程安全地拿一份当前 adaptive 模型引用 ─────────────────────────────
    with model_lock:
        model_ref = current_model          # 只在临界区做引用拷贝

    # === 关键：让两路输入与各自模型首层维度匹配 ============================
    X_base = _align_to_dim(X_scaled, baseline_in_dim)   # 基线模型固定 10 维
    X_adpt = _align_to_dim(X_scaled, model_ref.net[0].in_features)
        
        
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
    try:
        trigger_prod.send(RETRAIN_TOPIC, {"processed": batch_total})
    except Exception:
        pass
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


