# -*- coding: utf-8 -*-
import os
from kfp import dsl
from kfp.dsl import component, pipeline

# ----------------------------
# Global defaults (env fallbacks)
# ----------------------------
DEFAULT_S3_ENDPOINT   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
DEFAULT_S3_BUCKET     = os.getenv("S3_BUCKET",     "mlpipeline")
DEFAULT_S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
DEFAULT_S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
DEFAULT_S3_SECURE     = os.getenv("S3_SECURE",     "false")

DEFAULT_KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka.default.svc.cluster.local:9092")
DEFAULT_KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC",   "latencyTopic")

# Images
IMG_OFFLINE  = os.getenv("IMAGE_OFFLINE",  "hirschazer/offline:latest")
IMG_MONITOR  = os.getenv("IMAGE_MONITOR",  "hirschazer/monitor:latest")
IMG_PRODUCER = os.getenv("IMAGE_PRODUCER", "hirschazer/producer:latest")
IMG_INFER    = os.getenv("IMAGE_INFER",    "hirschazer/infer:latest")
IMG_PLOT     = os.getenv("IMAGE_PLOT",     "hirschazer/plot:latest")
IMG_RETRAIN  = os.getenv("IMAGE_RETRAIN",  "hirschazer/retrain:latest")


# ----------------------------
# Components
# ----------------------------
@component(base_image=IMG_OFFLINE)
def offline_training_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   DEFAULT_S3_ENDPOINT)
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     DEFAULT_S3_BUCKET)
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", DEFAULT_S3_ACCESS_KEY)
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", DEFAULT_S3_SECRET_KEY)
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     DEFAULT_S3_SECURE)
    subprocess.run(["python", "-m", "drst_inference.offline.train_offline"], check=True)


@component(base_image=IMG_PRODUCER)
def producer_op(
    kafka_topic: str = DEFAULT_KAFKA_TOPIC,
    interval_ms: int = 200,
    producer_stages: str = "",
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["KAFKA_BROKERS"] = os.getenv("KAFKA_BROKERS", DEFAULT_KAFKA_BROKERS)
    os.environ["KAFKA_TOPIC"]   = kafka_topic
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   DEFAULT_S3_ENDPOINT)
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     DEFAULT_S3_BUCKET)
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", DEFAULT_S3_ACCESS_KEY)
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", DEFAULT_S3_SECRET_KEY)
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     DEFAULT_S3_SECURE)
    if producer_stages:
        os.environ["PRODUCER_STAGES"] = producer_stages
    os.environ["INTERVAL_MS"] = str(interval_ms)
    subprocess.run(["python", "-m", "drst_inference.online.producer"], check=True)


@component(base_image=IMG_MONITOR)
def monitor_op(
    # —— monitor 专属参数（不会影响 retrain）——
    kafka_topic: str               = DEFAULT_KAFKA_TOPIC,
    drift_window: int              = 300,
    eval_stride: int               = 50,
    hist_bins: int                 = 64,
    idle_timeout_s: int            = 60,
    max_wall_secs: int             = 480,   # monitor 的 wall 上限（默认 480）
    js_quantiles: str              = "0.90,0.97,0.995",
    js_calib_samples: int          = 400,
    infer_pause_on_retrain: bool   = False,
    retrain_cooldown_s: int        = 10,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   DEFAULT_S3_ENDPOINT)
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     DEFAULT_S3_BUCKET)
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", DEFAULT_S3_ACCESS_KEY)
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", DEFAULT_S3_SECRET_KEY)
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     DEFAULT_S3_SECURE)
    os.environ["KAFKA_BROKERS"] = os.getenv("KAFKA_BROKERS", DEFAULT_KAFKA_BROKERS)
    os.environ["KAFKA_TOPIC"]   = kafka_topic

    os.environ["DRIFT_WINDOW"]  = str(drift_window)
    os.environ["EVAL_STRIDE"]   = str(eval_stride)
    os.environ["HIST_BINS"]     = str(hist_bins)
    os.environ["MONITOR_IDLE_TIMEOUT_S"] = str(idle_timeout_s)
    os.environ["MAX_WALL_SECS"] = str(max_wall_secs)  # ← 仅 monitor 使用

    os.environ["JS_QUANTILES"]     = js_quantiles
    os.environ["JS_CALIB_SAMPLES"] = str(js_calib_samples)
    os.environ["MONITOR_SIGNAL_INFER_PAUSE"] = "1" if infer_pause_on_retrain else "0"
    os.environ["RETRAIN_COOLDOWN_S"] = str(retrain_cooldown_s)
    subprocess.run(["python", "-m", "drst_drift.monitor"], check=True)


@component(base_image=IMG_RETRAIN)
def retrain_op(
    # —— retrain 专属参数（不影响 monitor）——
    watch: bool          = True,
    poll_interval_s: int = 2,
    max_wall_secs: int   = 120,   # retrain 的 wall 上限（默认 120）
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   DEFAULT_S3_ENDPOINT)
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     DEFAULT_S3_BUCKET)
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", DEFAULT_S3_ACCESS_KEY)
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", DEFAULT_S3_SECRET_KEY)
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     DEFAULT_S3_SECURE)
    os.environ["RETRAIN_WATCH"]   = "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"] = str(poll_interval_s)
    os.environ["MAX_WALL_SECS"]   = str(max_wall_secs)  # ← 仅 retrain 使用
    subprocess.run(["python", "-m", "drst_drift.dynamic_retrain"], check=True)


@component(base_image=IMG_INFER)
def infer_op(
    replica_id: int,
    kafka_topic: str   = DEFAULT_KAFKA_TOPIC,
    wait_retrain: bool = False,
    max_wall_secs: int = 480,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   DEFAULT_S3_ENDPOINT)
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     DEFAULT_S3_BUCKET)
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", DEFAULT_S3_ACCESS_KEY)
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", DEFAULT_S3_SECRET_KEY)
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     DEFAULT_S3_SECURE)
    os.environ["KAFKA_BROKERS"] = os.getenv("KAFKA_BROKERS", DEFAULT_KAFKA_BROKERS)
    os.environ["KAFKA_TOPIC"]   = kafka_topic
    os.environ["CONSUMER_WAIT_RETRAIN"] = "1" if wait_retrain else "0"
    os.environ["MAX_WALL_SECS"] = str(max_wall_secs)
    # 传给容器，便于资源文件命名固定为 infer1/2/3
    os.environ["INFER_REPLICA_ID"] = str(replica_id)
    subprocess.run(["python", "-m", "drst_inference.online.inference_consumer"], check=True)


@component(base_image=IMG_PLOT)
def plot_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   DEFAULT_S3_ENDPOINT)
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     DEFAULT_S3_BUCKET)
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", DEFAULT_S3_ACCESS_KEY)
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", DEFAULT_S3_SECRET_KEY)
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     DEFAULT_S3_SECURE)
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
    description="Drift monitoring + dynamic retraining + online inference (v2)"
)
def drift_stream_v2_pipeline(
    # General
    image: str                    = "unused",
    kafka_topic: str              = DEFAULT_KAFKA_TOPIC,

    # Producer
    producer_interval_ms: int     = 200,
    producer_stages: str          = "",

    # Monitor (只影响 monitor)
    drift_window: int             = 300,
    eval_stride: int              = 50,
    hist_bins: int                = 64,
    idle_timeout_s: int           = 60,
    max_wall_secs: int            = 480,  # monitor 的 wall

    js_quantiles: str             = "0.90,0.97,0.995",
    js_calib_samples: int         = 400,
    infer_pause_on_retrain: bool  = False,
    retrain_cooldown_s: int       = 10,

    # Retrain（只影响 retrain）
    retrain_watch: bool           = True,
    retrain_poll_interval_s: int  = 2,
    retrain_max_wall_secs: int    = 120,  # ← 关键：retrain 的 wall，默认 120

    # Infer
    infer_wait_retrain: bool      = False,
    infer_max_wall_secs: int      = 480,
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

    # 3) monitor（使用 monitor 的 max_wall_secs）
    monitor = monitor_op(
        kafka_topic=kafka_topic,
        drift_window=drift_window,
        eval_stride=eval_stride,
        hist_bins=hist_bins,
        idle_timeout_s=idle_timeout_s,
        max_wall_secs=max_wall_secs,          # ← 480 by default
        js_quantiles=js_quantiles,
        js_calib_samples=js_calib_samples,
        infer_pause_on_retrain=infer_pause_on_retrain,
        retrain_cooldown_s=retrain_cooldown_s,
    ).set_caching_options(False)
    monitor.after(offline)

    # 4) retrain（使用 retrain 的 retrain_max_wall_secs）
    retrain = retrain_op(
        watch=retrain_watch,
        poll_interval_s=retrain_poll_interval_s,
        max_wall_secs=retrain_max_wall_secs,  # ← 120 by default
    ).set_caching_options(False)
    retrain.after(offline)

    # 5) infer x3
    infer0 = infer_op(
        replica_id=0,
        kafka_topic=kafka_topic,
        wait_retrain=infer_wait_retrain,
        max_wall_secs=infer_max_wall_secs,
    ).set_caching_options(False)
    infer1 = infer_op(
        replica_id=1,
        kafka_topic=kafka_topic,
        wait_retrain=infer_wait_retrain,
        max_wall_secs=infer_max_wall_secs,
    ).set_caching_options(False)
    infer2 = infer_op(
        replica_id=2,
        kafka_topic=kafka_topic,
        wait_retrain=infer_wait_retrain,
        max_wall_secs=infer_max_wall_secs,
    ).set_caching_options(False)

    infer0.after(offline); infer1.after(offline); infer2.after(offline)

    # 6) plot（如不想被 retrain 阻塞，可把 retrain 从依赖里去掉）
    plot = plot_op().set_caching_options(False)
    plot.after(producer, monitor, retrain, infer0, infer1, infer2)


if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=drift_stream_v2_pipeline,
        package_path="drift_stream_v2.json"
    )
