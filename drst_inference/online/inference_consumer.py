#!/usr/bin/env python3
"""
drst_inference/online/inference_consumer.py

- 仅依赖本仓库的 drst_common.* 与 drst_inference.offline.*
- 从 Kafka 读取在线数据，走 baseline 与当前模型双路推理
- 定期根据 models/latest.txt 热更新当前模型（如有精度提升）
- 写出 results/<pod>_inference_trace.npz，供 plot 步骤使用
"""

from __future__ import annotations
import os
import io
import json
import time
import hashlib
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from kafka.errors import NoBrokersAvailable

# ---- 项目内统一工具 ----
from drst_common import config as _cfg  # for ACC_THR (optional)
from drst_common.config import (
    KAFKA_TOPIC, BATCH_SIZE, CONSUME_IDLE_S,
    MODEL_DIR, RESULT_DIR,
)
from drst_common.kafka_io import create_consumer, create_producer, partitions_count
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.minio_helper import save_bytes, s3
from drst_common.artefacts import (
    load_selected_feats, load_scaler,
    read_latest, load_model_by_key,
)
from drst_common.config import BUCKET  # for s3 key heads

# ===================== 运行参数（可用环境变量覆盖） =====================
IDLE_TIMEOUT_S    = int(os.getenv("IDLE_TIMEOUT_S", "300"))
RELOAD_INTERVAL_S = int(os.getenv("RELOAD_INTERVAL_S", "30"))
GAIN_THR_PP       = float(os.getenv("GAIN_THRESHOLD_PP", "0.01"))  # 新模型相对 baseline 的 acc@thr 提升（百分点）
RETRAIN_TOPIC     = os.getenv("RETRAIN_TOPIC", KAFKA_TOPIC + "_infer_count")

POD_NAME = os.getenv("HOSTNAME", "infer")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Accuracy threshold (relative error) — default 0.25; can be overridden via env ACC_THR
ACC_THR  = float(getattr(_cfg, "ACC_THR", float(os.getenv("ACC_THR", "0.25"))))
_thr_str = ("%.2f" % ACC_THR).rstrip("0").rstrip(".")

# ===================== 小工具 =====================
def _align_to_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    """把特征矩阵对齐到网络输入维度（截断或 0 填充）"""
    d = X.shape[1]
    if d == in_dim:
        return X
    if d > in_dim:
        return X[:, :in_dim]
    pad = np.zeros((X.shape[0], in_dim - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

def _load_json_key(key: str) -> Dict:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return json.loads(raw.decode())
    except Exception:
        return {}

def _pick_metric(metrics: Dict[str, float]) -> Tuple[float, float]:
    """
    From a retrain metrics json, pick (acc_new, acc_base) in a backward-compatible way.
    Prefer acc@<ACC_THR>; fallback to acc@0.15 if not present.
    """
    candidates = [f"acc@{_thr_str}", "acc@0.15", "acc"]
    base_candidates = [f"baseline_acc@{_thr_str}", "baseline_acc@0.15", "baseline_acc"]
    new_acc = None
    base_acc = None
    for k in candidates:
        if k in metrics:
            new_acc = float(metrics[k])
            break
    for k in base_candidates:
        if k in metrics:
            base_acc = float(metrics[k])
            break
    # If baseline missing, try infer from percentiles (not ideal) -> treat as 0
    return (new_acc if new_acc is not None else 0.0,
            base_acc if base_acc is not None else 0.0)

# ===================== 载入工件 =====================
print(f"[infer:{POD_NAME}] loading artefacts…")
SELECTED_FEATS = load_selected_feats()
SCALER = load_scaler()

# baseline 固定存在（由 offline 步骤写入）
baseline_model, baseline_raw = load_model_by_key("baseline_model.pt")
baseline_model = baseline_model.to(DEVICE).eval()
baseline_in_dim = baseline_model.net[0].in_features
print(f"[infer:{POD_NAME}] baseline in_dim = {baseline_in_dim}")

# 当前模型：latest.txt 指向；第一次找不到就回落到 baseline
_latest = read_latest()
_cur_key = (_latest[0] if _latest else "model.pt")
try:
    current_model, curr_raw = load_model_by_key(_cur_key)
    current_model = current_model.to(DEVICE).eval()
    print(f"[infer:{POD_NAME}] current model = {_cur_key}")
except Exception as e:
    print(f"[infer:{POD_NAME}] WARN load {_cur_key} failed: {e}; fallback -> baseline")
    current_model, curr_raw = baseline_model, baseline_raw
cur_sig = hashlib.md5(curr_raw).hexdigest()

# ===================== 热更新线程 =====================
_model_lock = torch.multiprocessing.get_context("spawn").RLock()
_model_loading_ms = 0.0

def _maybe_reload_model():
    """如 latest 指向了新模型，并且精度提升达到阈值，则热更新。"""
    global current_model, curr_raw, cur_sig, _model_loading_ms
    try:
        latest = read_latest()
        if not latest:
            return
        model_key, metrics_key, _ts = latest  # metrics_key 可能来自 offline 或 retrain
        # 对比签名
        head = s3.head_object(Bucket=BUCKET, Key=model_key)
        etag = head.get("ETag", "").strip('"')
        # 也读取真实 bytes 做哈希，避免某些后端 ETag 不稳定
        m, raw = load_model_by_key(model_key)
        sig = hashlib.md5(raw).hexdigest()
        if sig == cur_sig:
            return

        # 读取指标，判断是否需要切换（如果没有指标，就直接切换）
        metrics = _load_json_key(metrics_key)
        new_acc, base_acc = _pick_metric(metrics)
        gain_pp  = new_acc - base_acc

        if ("acc@0.15" in metrics or f"acc@{_thr_str}" in metrics) and (gain_pp < GAIN_THR_PP):
            print(f"[infer:{POD_NAME}] skip reload: Δ{gain_pp:+.3f}pp < {GAIN_THR_PP}")
            return

        t0 = time.perf_counter()
        mdl = m.to(DEVICE).eval()
        dt = (time.perf_counter() - t0) * 1000.0

        with _model_lock:
            current_model, curr_raw, cur_sig = mdl, raw, sig
            _model_loading_ms = round(dt, 3)

        log_metric(component="infer", event="hot_reload",
                   model_key=model_key, metrics_key=metrics_key,
                   gain_pp=round(gain_pp, 4), load_ms=_model_loading_ms)
        print(f"[infer:{POD_NAME}] hot reloaded -> {model_key} "
              f"(Δacc@{_thr_str}={gain_pp:+.3f}pp, load={_model_loading_ms}ms)")
    except Exception as e:
        print(f"[infer:{POD_NAME}] reload error: {e}")

def _reload_daemon():
    while True:
        time.sleep(RELOAD_INTERVAL_S)
        _maybe_reload_model()

import threading
threading.Thread(target=_reload_daemon, daemon=True).start()

# ===================== Kafka 消费循环 =====================
print(f"[infer:{POD_NAME}] connecting Kafka…")
consumer = create_consumer(KAFKA_TOPIC, group_id="cg-infer")
time.sleep(1.0)
num_parts = partitions_count(consumer, KAFKA_TOPIC)
print(f"[infer:{POD_NAME}] topic «{KAFKA_TOPIC}» partitions = {num_parts}")

# 就绪标记（可选）
save_bytes(f"{RESULT_DIR}/consumer_ready_{POD_NAME}.flag", b"", "text/plain")

producer = create_producer()
sentinel_seen = 0
last_data_ts = time.time()

# 轨迹累积（供绘图）
ts_hist:   List[float] = []
true_hist: List[float] = []
pred_b_hist: List[float] = []
pred_c_hist: List[float] = []

# 累计精度（@ACC_THR）
cum_correct = 0
cum_total   = 0

def _process_batch(rows: List[dict]):
    """对一批样本做推理、记录指标与轨迹。"""
    global cum_correct, cum_total

    # 预处理：抽取并按 SELECTED_FEATS 排列，缺失补 0
    X_raw = np.array(
        [[r["features"].get(c, 0.0) for c in SELECTED_FEATS] for r in rows],
        dtype=np.float32
    )
    labels = np.array([r.get("label", 0.0) for r in rows], dtype=np.float32)
    send_ts = [r.get("send_ts") for r in rows]

    X_sc = SCALER.transform(X_raw)

    with _model_lock:
        mdl = current_model
        in_dim_cur = mdl.net[0].in_features

    X_b = _align_to_dim(X_sc, baseline_in_dim)
    X_c = _align_to_dim(X_sc, in_dim_cur)

    t0 = time.perf_counter()
    with torch.no_grad():
        pb = baseline_model(torch.from_numpy(X_b).to(DEVICE)).cpu().numpy().ravel()
        pc = mdl            (torch.from_numpy(X_c).to(DEVICE)).cpu().numpy().ravel()
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0

    # 统计精度（@ACC_THR，相对误差）
    denom = np.maximum(np.abs(labels), 1e-8)
    err_c = np.abs(pc - labels) / denom
    batch_correct = int((err_c <= ACC_THR).sum())
    batch_total   = len(rows)
    cum_correct  += batch_correct
    cum_total    += batch_total
    cum_acc = (cum_correct / max(1, cum_total))

    # 分位数对比（相对误差）
    err_b = np.abs(pb - labels) / denom
    pct   = [50, 80, 90, 95, 99]
    q_b   = np.percentile(err_b, pct).round(3)
    q_c   = np.percentile(err_c, pct).round(3)

    log_metric(
        component="infer", event="batch_metrics",
        batch_size=batch_total, latency_ms=round(wall_ms, 3),
        **{f"cumulative_accuracy@{_thr_str}": round(cum_acc, 3)},
        base_p50=float(q_b[0]), base_p80=float(q_b[1]), base_p90=float(q_b[2]),
        base_p95=float(q_b[3]), base_p99=float(q_b[4]),
        adpt_p50=float(q_c[0]), adpt_p80=float(q_c[1]), adpt_p90=float(q_c[2]),
        adpt_p95=float(q_c[3]), adpt_p99=float(q_c[4]),
        model_loading_ms=_model_loading_ms
    )

    # 轨迹
    pred_b_hist.extend(pb.tolist())
    pred_c_hist.extend(pc.tolist())
    true_hist  .extend(labels.tolist())
    # 保存 epoch 秒时间戳
    for ts in send_ts:
        try:
            t = datetime.fromisoformat(str(ts).rstrip("Z")).timestamp()
        except Exception:
            t = time.time()
        ts_hist.append(float(t))

    # 回写处理数量（可选）
    try:
        producer.send(RETRAIN_TOPIC, {"processed": batch_total})
    except Exception:
        pass

print(f"[infer:{POD_NAME}] ready; consuming…")

batch_buf: List[dict] = []
BATCH = max(1, BATCH_SIZE)

while True:
    # poll 拉取一小段
    polled = consumer.poll(timeout_ms=1000)
    got_data = False

    for _, records in polled.items():
        for msg in records:
            v = msg.value
            if isinstance(v, dict) and v.get("producer_done"):
                sentinel_seen += 1
                print(f"[infer:{POD_NAME}] got sentinel {sentinel_seen}/{num_parts}")
                continue
            got_data = True
            last_data_ts = time.time()
            batch_buf.append(v)
            if len(batch_buf) >= BATCH:
                _process_batch(batch_buf); batch_buf.clear()

    # 无数据：检查空闲/终止条件
    if not got_data:
        # 剩余不足一个 batch 的也处理一下
        if batch_buf:
            _process_batch(batch_buf); batch_buf.clear()

        # 全部分区都收到终止哨兵 -> 退出
        if num_parts and sentinel_seen >= num_parts:
            print(f"[infer:{POD_NAME}] all sentinels received; exiting loop")
            break

        # 长时间空闲 -> 退出
        if (time.time() - last_data_ts) > IDLE_TIMEOUT_S:
            print(f"[infer:{POD_NAME}] idle > {IDLE_TIMEOUT_S}s; exiting loop")
            break

# ===================== 收尾：保存 trace =====================
arr_adj  = np.asarray(pred_c_hist,  np.float32)
arr_orig = np.asarray(pred_b_hist,  np.float32)
arr_true = np.asarray(true_hist,    np.float32)
arr_ts   = np.asarray(ts_hist,      np.float64)

local_npz = f"/tmp/{POD_NAME}_inference_trace.npz"
np.savez(local_npz, ts=arr_ts, pred_adj=arr_adj, pred_orig=arr_orig, true=arr_true)
with open(local_npz, "rb") as f:
    save_bytes(f"{RESULT_DIR}/{POD_NAME}_inference_trace.npz", f.read(), "application/octet-stream")
print(f"[infer:{POD_NAME}] trace saved: {len(arr_true)} samples")

sync_all_metrics_to_minio()
print(f"[infer:{POD_NAME}] done.")
