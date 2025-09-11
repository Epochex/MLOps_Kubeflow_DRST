# -*- coding: utf-8 -*-
"""
Kubeflow Pipeline (v2) — 简化版：用 idle 超时为主，wall 只作兜底（0=关闭）
- monitor 的 MAX_WALL_SECS 仅在传入 >0 时启用
- retrain 不再设置 MAX_WALL_SECS，始终跟随 monitor 的 monitor_done.flag 退出
- 方案A：在同一条流水线里集成 forecasting（训练 -> 解释 -> serve），与原先 offline/monitor/infer 并行但受 offline 产物约束
"""

import os
from kfp import dsl
from kfp.dsl import component, pipeline

# 镜像（可通过环境变量在编译时替换）
IMG_OFFLINE:   str = os.getenv("IMAGE_OFFLINE",   "hirschazer/offline:latest")
IMG_MONITOR:   str = os.getenv("IMAGE_MONITOR",   "hirschazer/monitor:latest")
IMG_PRODUCER:  str = os.getenv("IMAGE_PRODUCER",  "hirschazer/producer:latest")
IMG_INFER:     str = os.getenv("IMAGE_INFER",     "hirschazer/infer:latest")
IMG_PLOT:      str = os.getenv("IMAGE_PLOT",      "hirschazer/plot:latest")
IMG_RETRAIN:   str = os.getenv("IMAGE_RETRAIN",   "hirschazer/retrain:latest")
# 新增：forecast 通用镜像
IMG_FORECAST:  str = os.getenv("IMAGE_FORECAST",  "hirschazer/forecast:latest")

# ----------------------------
# Components
# ----------------------------
@component(base_image=IMG_OFFLINE)
def offline_training_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    subprocess.run(["python", "-m", "drst_inference.offline.train_offline"], check=True)

@component(base_image=IMG_PRODUCER)
def producer_op(
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 200,
    producer_stages: str = "",
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if kafka_topic:
        os.environ["KAFKA_TOPIC"] = str(kafka_topic)
    os.environ["INTERVAL_MS"] = str(int(interval_ms))
    if producer_stages:
        os.environ["PRODUCER_STAGES"] = producer_stages
    subprocess.run(["python", "-m", "drst_inference.online.producer"], check=True)

@component(base_image=IMG_MONITOR)
def monitor_op(max_wall_secs: int = 0) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if int(max_wall_secs) > 0:
        os.environ["MAX_WALL_SECS"] = str(int(max_wall_secs))
    subprocess.run(["python", "-m", "drst_drift.monitor"], check=True)

@component(base_image=IMG_RETRAIN)
def retrain_op(watch: bool = True, poll_interval_s: int = 2) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["RETRAIN_WATCH"]   = "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"] = str(int(poll_interval_s))
    subprocess.run(["python", "-m", "drst_drift.dynamic_retrain"], check=True)

@component(base_image=IMG_INFER)
def infer_op(replica_id: int, kafka_topic: str = "latencyTopic") -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if kafka_topic:
        os.environ["KAFKA_TOPIC"] = str(kafka_topic)
    os.environ["INFER_REPLICA_ID"] = str(int(replica_id))
    subprocess.run(["python", "-m", "drst_inference.online.inference_consumer"], check=True)

@component(base_image=IMG_PLOT)
def plot_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    try:
        subprocess.run(["python", "-m", "drst_inference.plotting.plot_final"], check=True)
    except Exception as e:
        print(f"[plot_final] skipped: {e}", flush=True)
    try:
        subprocess.run(["python", "experiments/kubeflow/summarize_resources.py"], check=True)
    except Exception as e:
        print(f"[plot_op] summarize_resources.py skipped: {e}", flush=True)
    try:
        subprocess.run(["python", "-m", "drst_inference.plotting.plot_report"], check=True)
    except Exception as e:
        print(f"[plot_report] skipped: {e}", flush=True)
    # 注：不再单独生成 drst_forecasting.report

# ----------------------------
# Forecasting components（直接调用核心模块）
# ----------------------------
@component(base_image=IMG_FORECAST)
def forecast_train_op(
    lookback: int = 48,
    horizon: int = 12,
    epochs: int = 20,
    patience: int = 5,
    hidden: int = 64,
    layers: int = 1,
    batch_size: int = 64,
    take_last: int = 0,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["FORECAST_LOOKBACK"]  = str(int(lookback))
    os.environ["FORECAST_HORIZON"]   = str(int(horizon))
    os.environ["FORECAST_EPOCHS"]    = str(int(epochs))
    os.environ["FORECAST_PATIENCE"]  = str(int(patience))
    os.environ["FORECAST_HIDDEN"]    = str(int(hidden))
    os.environ["FORECAST_LAYERS"]    = str(int(layers))
    os.environ["FORECAST_BS"]        = str(int(batch_size))
    os.environ["FORECAST_TAKE_LAST"] = str(int(take_last))
    subprocess.run(["python", "-m", "drst_forecasting.train_benchmark"], check=True)

@component(base_image=IMG_FORECAST)
def forecast_explain_op(
    lookback: int = 48,
    horizon: int = 12,
    shap_n: int = 256,
    hidden: int = 64,
    layers: int = 1,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["FORECAST_LOOKBACK"] = str(int(lookback))
    os.environ["FORECAST_HORIZON"]  = str(int(horizon))
    os.environ["FORECAST_SHAP_N"]   = str(int(shap_n))
    os.environ["FORECAST_HIDDEN"]   = str(int(hidden))
    os.environ["FORECAST_LAYERS"]   = str(int(layers))
    subprocess.run(["python", "-m", "drst_forecasting.explain"], check=True)

@component(base_image=IMG_FORECAST)
def forecast_serve_op(
    lookback: int = 48,
    horizon: int = 12,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["FORECAST_LOOKBACK"] = str(int(lookback))
    os.environ["FORECAST_HORIZON"]  = str(int(horizon))
    subprocess.run(["python", "-m", "drst_forecasting.serve_forecaster"], check=True)

# ----------------------------
# Pipeline
# ----------------------------
@pipeline(
    name="drift-stream-v2",
    description="Drift monitoring + dynamic retraining + online inference (v2) — with forecasting (Plan A, slim)"
)
def drift_stream_v2_pipeline(
    kafka_topic: str = "latencyTopic",
    producer_interval_ms: int = 200,
    producer_stages: str = "",
    monitor_max_wall_secs: int = 0,
    fc_lookback: int = 48,
    fc_horizon:  int = 12,
) -> None:
    # 1) offline
    offline = offline_training_op().set_caching_options(False)

    # 2) producer
    producer = producer_op(
        kafka_topic=kafka_topic,
        interval_ms=producer_interval_ms,
        producer_stages=producer_stages,
    ).set_caching_options(False)
    producer.after(offline)

    # 3) monitor
    monitor = monitor_op(max_wall_secs=monitor_max_wall_secs).set_caching_options(False)
    monitor.after(offline)

    # 4) retrain watcher
    retrain = retrain_op(watch=True, poll_interval_s=2).set_caching_options(False)
    retrain.after(offline)

    # 5) infer x3
    infer0 = infer_op(replica_id=0, kafka_topic=kafka_topic).set_caching_options(False)
    infer1 = infer_op(replica_id=1, kafka_topic=kafka_topic).set_caching_options(False)
    infer2 = infer_op(replica_id=2, kafka_topic=kafka_topic).set_caching_options(False)
    infer0.after(offline); infer1.after(offline); infer2.after(offline)

    # 6) forecasting（受 offline 产物约束，因此也 .after(offline)）
    fc_train = forecast_train_op(lookback=fc_lookback, horizon=fc_horizon).set_caching_options(False)
    fc_train.after(offline)

    fc_explain = forecast_explain_op(lookback=fc_lookback, horizon=fc_horizon).set_caching_options(False)
    fc_explain.after(fc_train)

    fc_serve = forecast_serve_op(lookback=fc_lookback, horizon=fc_horizon).set_caching_options(False)
    fc_serve.after(fc_train)

    # 7) plot 汇总
    plot = plot_op().set_caching_options(False)
    plot.after(producer, monitor, retrain, infer0, infer1, infer2, fc_explain, fc_serve)

if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=drift_stream_v2_pipeline,
        package_path="drift_stream_v2.json"
    )
