# -*- coding: utf-8 -*-
"""
Kubeflow Pipeline (v2) — 简化版：用 idle 超时为主，wall 只作兜底（0=关闭）
- monitor 的 MAX_WALL_SECS 仅在传入 >0 时启用
- retrain 不再设置 MAX_WALL_SECS，始终跟随 monitor 的 monitor_done.flag 退出
"""

import os
from kfp import dsl
from kfp.dsl import component, pipeline

# 镜像（可通过环境变量在编译时替换）
IMG_OFFLINE:  str = os.getenv("IMAGE_OFFLINE",  "hirschazer/offline:latest")
IMG_MONITOR:  str = os.getenv("IMAGE_MONITOR",  "hirschazer/monitor:latest")
IMG_PRODUCER: str = os.getenv("IMAGE_PRODUCER", "hirschazer/producer:latest")
IMG_INFER:    str = os.getenv("IMAGE_INFER",    "hirschazer/infer:latest")
IMG_PLOT:     str = os.getenv("IMAGE_PLOT",     "hirschazer/plot:latest")
IMG_RETRAIN:  str = os.getenv("IMAGE_RETRAIN",  "hirschazer/retrain:latest")


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
def monitor_op(
    max_wall_secs: int = 0,   # 默认 0=关闭墙时长，只按 idle 超时退出
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # 仅当 >0 时才设置，避免无意中启用墙时长
    if int(max_wall_secs) > 0:
        os.environ["MAX_WALL_SECS"] = str(int(max_wall_secs))
    subprocess.run(["python", "-m", "drst_drift.monitor"], check=True)


@component(base_image=IMG_RETRAIN)
def retrain_op(
    watch: bool = True,
    poll_interval_s: int = 2,
) -> None:
    """
    retrain watcher：不再设置 MAX_WALL_SECS，默认无限守护，
    仅处理锁并“跟随 monitor_done.flag”退出（逻辑在 dynamic_retrain.py 里）。
    """
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["RETRAIN_WATCH"]   = "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"] = str(int(poll_interval_s))
    # 刻意不设置 MAX_WALL_SECS：让 watcher 不受墙时长影响
    subprocess.run(["python", "-m", "drst_drift.dynamic_retrain"], check=True)


@component(base_image=IMG_INFER)
def infer_op(
    replica_id: int,
    kafka_topic: str = "latencyTopic",
) -> None:
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


# ----------------------------
# Pipeline
# ----------------------------
@pipeline(
    name="drift-stream-v2",
    description="Drift monitoring + dynamic retraining + online inference (v2) — simplified"
)
def drift_stream_v2_pipeline(
    kafka_topic: str = "latencyTopic",

    # Producer 速率
    producer_interval_ms: int = 200,
    producer_stages: str = "",

    # 仅 monitor 使用的墙时长（0=关闭；建议主要用 idle 超时）
    monitor_max_wall_secs: int = 0,
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
    monitor = monitor_op(
        max_wall_secs=monitor_max_wall_secs,
    ).set_caching_options(False)
    monitor.after(offline)

    # 4) retrain（无限守护，跟随 monitor_done 退出）
    retrain = retrain_op(
        watch=True,
        poll_interval_s=2,
    ).set_caching_options(False)
    retrain.after(offline)

    # 5) infer x3
    infer0 = infer_op(replica_id=0, kafka_topic=kafka_topic).set_caching_options(False)
    infer1 = infer_op(replica_id=1, kafka_topic=kafka_topic).set_caching_options(False)
    infer2 = infer_op(replica_id=2, kafka_topic=kafka_topic).set_caching_options(False)
    infer0.after(offline); infer1.after(offline); infer2.after(offline)

    # 6) plot
    plot = plot_op().set_caching_options(False)
    plot.after(producer, monitor, retrain, infer0, infer1, infer2)


if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=drift_stream_v2_pipeline,
        package_path="drift_stream_v2.json"
    )
