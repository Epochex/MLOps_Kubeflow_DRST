#!/usr/bin/env python3
# experiments/kubeflow/pipeline.py

from __future__ import annotations

from kfp import compiler
from kfp.v2.dsl import component, pipeline, OutputPath

# ======== Images ========
IMAGE_OFFLINE  = "hirschazer/offline:latest"
IMAGE_MONITOR  = "hirschazer/monitor:latest"
IMAGE_PRODUCER = "hirschazer/producer:latest"
IMAGE_INFER    = "hirschazer/infer:latest"
IMAGE_PLOT     = "hirschazer/plot:latest"

# ======== Components ========

@component(base_image=IMAGE_OFFLINE)
def offline_op(
    output_metadata: OutputPath(str),
    train_minio_key: str = "datasets_old/offline/combined.csv",
):
    """Phase-1 Offline training: writes model artifacts into models/ (baseline, model.pt, metrics, latest.txt)"""
    import os, subprocess
    os.environ["TRAIN_MINIO_KEY"] = train_minio_key
    subprocess.run(["python", "drst_inference/offline/train_offline.py"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")

@component(base_image=IMAGE_OFFLINE)
def monitor_op(
    output_metadata: OutputPath(str),
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
    import os, subprocess, shlex, signal, time
    env = os.environ.copy()
    env["KAFKA_TOPIC"]            = kafka_topic
    env["IDLE_TIMEOUT_S"]         = str(idle_timeout_s)
    env["HIST_BINS"]              = str(hist_bins)
    env["DRIFT_WINDOW"]           = str(drift_window)
    env["EVAL_STRIDE"]            = str(eval_stride)
    env["JS_CALIB_SAMPLES"]       = str(js_calib_samples)
    env["JS_QUANTILES"]           = js_quantiles
    env["BASELINE_REFRESH_MODE"]  = baseline_refresh_mode

    cmd = f"python drst_drift/monitor.py"
    p = subprocess.Popen(shlex.split(cmd), env=env)
    t0 = time.time()
    while p.poll() is None:
        if time.time() - t0 > max_wall_secs:
            p.kill()
            break
        time.sleep(1)
    with open(output_metadata, "w") as f:
        f.write("{}")

@component(base_image=IMAGE_OFFLINE)
def producer_op(
    output_metadata: OutputPath(str),
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 100,
    s1_n: int = 3000,
    s2_n: int = 1000,
    s3_n: int = 1000,
):
    """Phase-3 Kafka producer (send data in stages, then broadcast termination sentinels by partition)"""
    import os, subprocess
    env = os.environ.copy()
    env["KAFKA_TOPIC"]         = kafka_topic
    env["PRODUCE_INTERVAL_MS"] = str(interval_ms)
    env["STAGE1_N"]            = str(s1_n)
    env["STAGE2_N"]            = str(s2_n)
    env["STAGE3_N"]            = str(s3_n)

    subprocess.run(["python", "drst_inference/online/producer.py"], check=True, env=env)
    with open(output_metadata, "w") as f:
        f.write("{}")

@component(base_image=IMAGE_OFFLINE)
def inference_op(
    output_metadata: OutputPath(str),
    kafka_topic: str = "latencyTopic",
    idle_timeout_s: int = 60,
    reload_interval_s: int = 30,
    instance_id: int = 0,
    max_wall_secs: int = 480,
):
    """Phase-4 Online inference consumer (dual-path baseline+adaptive with hot-reload)"""
    import os, subprocess, shlex, time
    env = os.environ.copy()
    env["KAFKA_TOPIC"]       = kafka_topic
    env["IDLE_TIMEOUT_S"]    = str(idle_timeout_s)
    env["RELOAD_INTERVAL_S"] = str(reload_interval_s)
    env["HOSTNAME"]          = f"infer-{instance_id}"

    cmd = "python drst_inference/online/inference_consumer.py"
    p = subprocess.Popen(shlex.split(cmd), env=env)
    t0 = time.time()
    while p.poll() is None:
        if time.time() - t0 > max_wall_secs:
            p.kill()
            break
        time.sleep(1)
    with open(output_metadata, "w") as f:
        f.write("{}")

@component(base_image=IMAGE_OFFLINE)
def plot_op(output_metadata: OutputPath(str)):
    """Phase-5 Summary plotting and reporting (reads *_inference_trace.npz; generates PNG + report.md to MinIO)"""
    import subprocess
    subprocess.run(["python", "drst_inference/plotting/plot_final.py"],  check=True)
    subprocess.run(["python", "drst_inference/plotting/plot_report.py"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")

# ======== Pipeline ========
@pipeline(name="drift-stream-v2")
def drift_stream(
    image: str = IMAGE_OFFLINE,
    offline_key: str = "datasets_old/offline/combined.csv",
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 100,
    s1_n: int = 3000, s2_n: int = 1000, s3_n: int = 1000,
    idle_timeout_s: int = 60,
    max_wall_secs: int = 480,
):
    """
    Topology:
      offline → (monitor || producer || infer×3) → plot
    Behavior highlights:
      • Monitor uses bootstrap-calibrated JSD thresholds and refreshes baseline after successful retraining;
      • Components auto-stop on idle; producer sends partitioned sentinels; plot summarizes artifacts.
    """
    off = offline_op(train_minio_key=offline_key).set_display_name("offline-training").set_caching_options(False)

    final_plot = plot_op().set_display_name("plot-and-report").set_caching_options(False)

    from kfp.v2.dsl import ExitHandler
    with ExitHandler(final_plot):
        mon = monitor_op(
            kafka_topic=kafka_topic,
            idle_timeout_s=idle_timeout_s,
            max_wall_secs=max_wall_secs,
        ).after(off).set_display_name("drift-monitor").set_caching_options(False)

        prod = producer_op(
            kafka_topic=kafka_topic,
            interval_ms=interval_ms,
            s1_n=s1_n, s2_n=s2_n, s3_n=s3_n,
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

        final_plot.after(prod, mon, *consumers)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v2.yaml",
    )
    print(" drift_demo_v2.yaml generated successfully")
