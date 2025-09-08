# -*- coding: utf-8 -*-
import os
from kfp import dsl
from kfp.dsl import component, pipeline

# 这些模块级“默认值”仅用于编译期（比如 base_image、管道参数等）；
# ⚠️ 组件函数体内不要再引用它们，避免被 KFP 裁剪后 NameError。
DEFAULT_S3_ENDPOINT   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
DEFAULT_S3_BUCKET     = os.getenv("S3_BUCKET",     "mlpipeline")
DEFAULT_S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
DEFAULT_S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
DEFAULT_S3_SECURE     = os.getenv("S3_SECURE",     "false")

DEFAULT_KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka.default.svc.cluster.local:9092")
DEFAULT_KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC",   "latencyTopic")

# 为每个组件绑定各自镜像（与 docker/build_and_push.sh 对齐）
IMG_OFFLINE  = os.getenv("IMAGE_OFFLINE",  "hirschazer/offline:latest")
IMG_MONITOR  = os.getenv("IMAGE_MONITOR",  "hirschazer/monitor:latest")
IMG_PRODUCER = os.getenv("IMAGE_PRODUCER", "hirschazer/producer:latest")
IMG_INFER    = os.getenv("IMAGE_INFER",    "hirschazer/infer:latest")
IMG_PLOT     = os.getenv("IMAGE_PLOT",     "hirschazer/plot:latest")
IMG_RETRAIN  = os.getenv("IMAGE_RETRAIN",  "hirschazer/retrain:latest")


# --------- Components ---------
@component(base_image=IMG_OFFLINE)
def offline_training_op() -> None:
    """线下初训（写 feature_cols.json / baseline_model.pt / latest 指针）"""
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # 直接使用字面量默认值作为 getenv 的后备，避免 NameError
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     "mlpipeline")
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", "minio")
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", "minio123")
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     "false")
    subprocess.run(["python", "-m", "drst_inference.offline.train_offline"], check=True)


@component(base_image=IMG_PRODUCER)
def producer_op(
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 200,
    producer_stages: str = "",
) -> None:
    """Kafka producer（限流）"""
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # Kafka
    os.environ["KAFKA_BROKERS"] = os.getenv("KAFKA_BROKERS", "kafka.default.svc.cluster.local:9092")
    os.environ["KAFKA_TOPIC"]   = kafka_topic
    # S3（producer 也要读 CSV）
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     "mlpipeline")
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", "minio")
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", "minio123")
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     "false")
    # 节流
    if producer_stages:
        os.environ["PRODUCER_STAGES"] = producer_stages
    os.environ["INTERVAL_MS"] = str(interval_ms)
    subprocess.run(["python", "-m", "drst_inference.online.producer"], check=True)


@component(base_image=IMG_MONITOR)
def monitor_op(
    kafka_topic: str               = "latencyTopic",
    drift_window: int              = 300,
    eval_stride: int               = 50,
    hist_bins: int                 = 64,
    idle_timeout_s: int            = 60,
    max_wall_secs: int             = 480,
    js_quantiles: str              = "0.90,0.97,0.995",
    js_calib_samples: int          = 400,
    infer_pause_on_retrain: bool   = False,
) -> None:
    """JS 漂移监控，触发 retrain"""
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # S3
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     "mlpipeline")
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", "minio")
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", "minio123")
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     "false")
    # Kafka
    os.environ["KAFKA_BROKERS"] = os.getenv("KAFKA_BROKERS", "kafka.default.svc.cluster.local:9092")
    os.environ["KAFKA_TOPIC"]   = kafka_topic
    # 监控参数
    os.environ["DRIFT_WINDOW"]  = str(drift_window)
    os.environ["EVAL_STRIDE"]   = str(eval_stride)
    os.environ["HIST_BINS"]     = str(hist_bins)
    os.environ["MONITOR_IDLE_TIMEOUT_S"] = str(idle_timeout_s)
    os.environ["MAX_WALL_SECS"] = str(max_wall_secs)
    os.environ["JS_QUANTILES"]     = js_quantiles
    os.environ["JS_CALIB_SAMPLES"] = str(js_calib_samples)
    os.environ["MONITOR_SIGNAL_INFER_PAUSE"] = "1" if infer_pause_on_retrain else "0"
    subprocess.run(["python", "-m", "drst_drift.monitor"], check=True)


@component(base_image=IMG_RETRAIN)
def retrain_op(
    watch: bool          = True,
    poll_interval_s: int = 2,
    max_wall_secs: int   = 480,
) -> None:
    """动态重训 watcher：发现 lock 就执行一次 dynamic_retrain；持续到 max_wall_secs"""
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # S3 for dynamic_retrain
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     "mlpipeline")
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", "minio")
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", "minio123")
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     "false")

    os.environ["RETRAIN_WATCH"]   = "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"] = str(poll_interval_s)
    os.environ["MAX_WALL_SECS"]   = str(max_wall_secs)
    subprocess.run(["python", "-m", "drst_drift.dynamic_retrain"], check=True)


@component(base_image=IMG_INFER)
def infer_op(
    replica_id: int,
    kafka_topic: str   = "latencyTopic",
    wait_retrain: bool = False,
    max_wall_secs: int = 480,
) -> None:
    """在线推理副本"""
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # S3
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     "mlpipeline")
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", "minio")
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", "minio123")
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     "false")
    # Kafka
    os.environ["KAFKA_BROKERS"] = os.getenv("KAFKA_BROKERS", "kafka.default.svc.cluster.local:9092")
    os.environ["KAFKA_TOPIC"]   = kafka_topic
    # 其它
    os.environ["CONSUMER_WAIT_RETRAIN"] = "1" if wait_retrain else "0"
    os.environ["MAX_WALL_SECS"] = str(max_wall_secs)
    subprocess.run(["python", "-m", "drst_inference.online.inference_consumer"], check=True)


@component(base_image=IMG_PLOT)
def plot_op() -> None:
    """汇总绘图；失败不影响主流程"""
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    # S3
    os.environ["S3_ENDPOINT"]   = os.getenv("S3_ENDPOINT",   "http://minio-service.kubeflow.svc.cluster.local:9000")
    os.environ["S3_BUCKET"]     = os.getenv("S3_BUCKET",     "mlpipeline")
    os.environ["S3_ACCESS_KEY"] = os.getenv("S3_ACCESS_KEY", "minio")
    os.environ["S3_SECRET_KEY"] = os.getenv("S3_SECRET_KEY", "minio123")
    os.environ["S3_SECURE"]     = os.getenv("S3_SECURE",     "false")
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


# ------------------------ Pipeline ------------------------
@pipeline(
    name="drift-stream-v2",
    description="Drift monitoring + dynamic retraining + online inference (v2)"
)
def drift_stream_v2_pipeline(
    image: str                    = "unused",  # 保留历史参数位；现已改为按组件镜像
    kafka_topic: str              = DEFAULT_KAFKA_TOPIC,
    producer_interval_ms: int     = 200,
    producer_stages: str          = "",
    drift_window: int             = 300,
    eval_stride: int              = 50,
    hist_bins: int                = 64,
    idle_timeout_s: int           = 60,
    max_wall_secs: int            = 480,
    js_quantiles: str             = "0.90,0.97,0.995",
    js_calib_samples: int         = 400,
    infer_pause_on_retrain: bool  = False,
    retrain_watch: bool           = True,
    retrain_poll_interval_s: int  = 2,
    retrain_max_wall_secs: int    = 480,
    infer_wait_retrain: bool      = False,
    infer_max_wall_secs: int      = 480,
) -> None:
    offline = offline_training_op().set_caching_options(False)

    producer = producer_op(
        kafka_topic=kafka_topic,
        interval_ms=producer_interval_ms,
        producer_stages=producer_stages
    ).set_caching_options(False)
    producer.after(offline)

    monitor = monitor_op(
        kafka_topic=kafka_topic,
        drift_window=drift_window,
        eval_stride=eval_stride,
        hist_bins=hist_bins,
        idle_timeout_s=idle_timeout_s,
        max_wall_secs=max_wall_secs,
        js_quantiles=js_quantiles,
        js_calib_samples=js_calib_samples,
        infer_pause_on_retrain=infer_pause_on_retrain
    ).set_caching_options(False)
    monitor.after(offline)

    retrain = retrain_op(
        watch=retrain_watch,
        poll_interval_s=retrain_poll_interval_s,
        max_wall_secs=retrain_max_wall_secs
    ).set_caching_options(False)
    retrain.after(offline)

    infer0 = infer_op(replica_id=0, kafka_topic=kafka_topic,
                      wait_retrain=infer_wait_retrain, max_wall_secs=infer_max_wall_secs).set_caching_options(False)
    infer1 = infer_op(replica_id=1, kafka_topic=kafka_topic,
                      wait_retrain=infer_wait_retrain, max_wall_secs=infer_max_wall_secs).set_caching_options(False)
    infer2 = infer_op(replica_id=2, kafka_topic=kafka_topic,
                      wait_retrain=infer_wait_retrain, max_wall_secs=infer_max_wall_secs).set_caching_options(False)
    infer0.after(offline); infer1.after(offline); infer2.after(offline)

    plot = plot_op().set_caching_options(False)
    plot.after(producer, monitor, retrain, infer0, infer1, infer2)


if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=drift_stream_v2_pipeline,
        package_path="drift_stream_v2.json"
    )
