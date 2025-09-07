#!/usr/bin/env python3
# experiments/kubeflow/pipeline.py

from kfp import dsl, compiler

# ======== Images (与你 push 的镜像标签一致) ========
IMAGE_OFFLINE  = "hirschazer/offline:latest"
IMAGE_MONITOR  = "hirschazer/monitor:latest"
IMAGE_PRODUCER = "hirschazer/producer:latest"
IMAGE_INFER    = "hirschazer/infer:latest"
IMAGE_PLOT     = "hirschazer/plot:latest"

# ======== Components ========

@dsl.component(base_image=IMAGE_OFFLINE)
def offline_op(
    train_minio_key: str = "datasets/combined.csv",
):
    """Phase-1 Offline training"""
    import os, sys, subprocess
    app_dir = os.environ.get("APP_DIR", "/app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    env = os.environ.copy()
    env["TRAIN_MINIO_KEY"] = train_minio_key

    subprocess.run(
        ["python", "-m", "drst_inference.offline.train_offline"],
        check=True,
        cwd=app_dir,
        env=env,
    )


@dsl.component(base_image=IMAGE_MONITOR)
def monitor_op(
    kafka_topic: str = "latencyTopic",
    idle_timeout_s: int = 60,
    max_wall_secs: int = 480,
    # Drift/JSD settings (calibrated thresholds)
    js_calib_samples: int = 400,
    js_quantiles: str = "0.90,0.97,0.995",   # A,B,C quantiles
    baseline_refresh_mode: str = "on_retrain",# on_retrain|never
    hist_bins: int = 64,
    drift_window: int = 300,
    eval_stride: int = 50,
):
    """Phase-2 Drift monitoring (sliding-window JSD with bootstrap-calibrated thresholds)"""
    import os, sys, subprocess, time
    app_dir = os.environ.get("APP_DIR", "/app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    env = os.environ.copy()
    env["KAFKA_TOPIC"]           = kafka_topic
    env["IDLE_TIMEOUT_S"]        = str(idle_timeout_s)
    env["HIST_BINS"]             = str(hist_bins)
    env["DRIFT_WINDOW"]          = str(drift_window)
    env["EVAL_STRIDE"]           = str(eval_stride)
    env["JS_CALIB_SAMPLES"]      = str(js_calib_samples)
    env["JS_QUANTILES"]          = js_quantiles
    env["BASELINE_REFRESH_MODE"] = baseline_refresh_mode

    p = subprocess.Popen(
        ["python", "-m", "drst_drift.monitor"],
        cwd=app_dir,
        env=env,
    )
    t0 = time.time()
    try:
        while p.poll() is None:
            if time.time() - t0 > max_wall_secs:
                p.kill()
                break
            time.sleep(1)
    finally:
        try:
            p.terminate()
        except Exception:
            pass


@dsl.component(base_image=IMAGE_PRODUCER)
def producer_op(
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 100,
    producer_stages: str = "",   # 可选：覆盖 config.PRODUCER_STAGES
):
    """Phase-3 Kafka producer"""
    import os, sys, subprocess
    app_dir = os.environ.get("APP_DIR", "/app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    env = os.environ.copy()
    env["KAFKA_TOPIC"]         = kafka_topic
    env["PRODUCE_INTERVAL_MS"] = str(interval_ms)
    # 严格轮询各分区（最小增改，仅此一行）
    env["PRODUCER_PARTITION_MODE"] = "rr"
    if producer_stages:
        env["PRODUCER_STAGES"] = producer_stages

    subprocess.run(
        ["python", "-m", "drst_inference.online.producer"],
        check=True,
        cwd=app_dir,
        env=env,
    )


@dsl.component(base_image=IMAGE_INFER)
def inference_op(
    kafka_topic: str = "latencyTopic",
    idle_timeout_s: int = 60,
    reload_interval_s: int = 30,
    instance_id: int = 0,
    max_wall_secs: int = 480,
):
    """Phase-4 Online inference consumer (dual-path baseline+adaptive with hot-reload)"""
    import os, sys, subprocess, time
    app_dir = os.environ.get("APP_DIR", "/app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    env = os.environ.copy()
    env["KAFKA_TOPIC"]        = kafka_topic
    env["IDLE_TIMEOUT_S"]     = str(idle_timeout_s)
    env["RELOAD_INTERVAL_S"]  = str(reload_interval_s)
    env["HOSTNAME"]           = f"infer-{instance_id}"

    p = subprocess.Popen(
        ["python", "-m", "drst_inference.online.inference_consumer"],
        cwd=app_dir,
        env=env,
    )
    t0 = time.time()
    try:
        while p.poll() is None:
            if time.time() - t0 > max_wall_secs:
                p.kill()
                break
            time.sleep(1)
    finally:
        try:
            p.terminate()
        except Exception:
            pass


@dsl.component(base_image=IMAGE_PLOT)
def plot_op():
    """Phase-5 Summary plotting and reporting"""
    import os, sys, subprocess
    app_dir = os.environ.get("APP_DIR", "/app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    subprocess.run(
        ["python", "-m", "drst_inference.plotting.plot_final"],
        check=True,
        cwd=app_dir,
    )
    subprocess.run(
        ["python", "-m", "drst_inference.plotting.plot_report"],
        check=True,
        cwd=app_dir,
    )

    # （可选）若镜像里带有 experiments/summarize_resources.py，则顺手产出资源汇总；
    # 若文件不存在或失败，不影响流水线成功。
    try:
        subprocess.run(
            ["python", "experiments/summarize_resources.py"],
            check=True,
            cwd=app_dir,
        )
    except Exception as e:
        print("[plot_op] summarize_resources.py skipped:", e)

# ======== Pipeline ========
@dsl.pipeline(name="drift-stream-v2")
def drift_stream(
    image: str = IMAGE_OFFLINE,            # 未使用，仅保留兼容
    offline_key: str = "datasets/combined.csv",
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 100,
    producer_stages: str = "",
    idle_timeout_s: int = 60,
    max_wall_secs: int = 480,
):
    """
    Topology:
      offline → (monitor || producer || infer×3) → plot
    """
    off = offline_op(train_minio_key=offline_key).set_display_name("offline-training").set_caching_options(False)

    mon = monitor_op(
        kafka_topic=kafka_topic,
        idle_timeout_s=idle_timeout_s,
        max_wall_secs=max_wall_secs,
    ).after(off).set_display_name("drift-monitor").set_caching_options(False)

    prod = producer_op(
        kafka_topic=kafka_topic,
        interval_ms=interval_ms,
        producer_stages=producer_stages,
    ).after(off).set_display_name("kafka-producer").set_caching_options(False)

    consumers = []
    for i in range(3):
        c = inference_op(
            kafka_topic=kafka_topic,
            idle_timeout_s=idle_timeout_s,
            instance_id=i,
            max_wall_secs=max_wall_secs,
        ).after(off).set_display_name(f"online-infer-{i}").set_caching_options(False)
        consumers.append(c)

    final_plot = plot_op().set_display_name("plot-and-report").set_caching_options(False)
    final_plot.after(prod, mon, *consumers)

    # 统一使用 in-cluster MinIO，避免 ingress 依赖
    for t in [off, mon, prod, *consumers, final_plot]:
        t.set_env_variable(name="MINIO_ACCESS_MODE", value="cluster")
        t.set_env_variable(name="MINIO_ENDPOINT",   value="minio-service.kubeflow.svc.cluster.local:9000")
        t.set_env_variable(name="MINIO_SCHEME",     value="http")
        t.set_env_variable(name="MINIO_BUCKET",     value="onvm-demo2")

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v2.yaml",
    )
    print(" drift_demo_v2.yaml generated successfully")
