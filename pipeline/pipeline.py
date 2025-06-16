# pipeline/pipeline.py

from kfp import compiler
from kfp.v2.dsl import pipeline, component, OutputPath

# ────────────────────────────────────────────────────────────────
# 1. 定义五个组件：在函数内部显式 import subprocess
# ────────────────────────────────────────────────────────────────

@component(base_image="hirschazer/offline:latest")
def offline_op(output_metadata: OutputPath(str)):
    import subprocess
    """Phase-1 离线基线训练"""
    subprocess.run(["python", "-m", "ml.train_offline"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")


@component(base_image="hirschazer/monitor:latest")
def monitor_op(output_metadata: OutputPath(str)):
    import subprocess
    """Phase-2 Drift 监控"""
    subprocess.run(["python", "-m", "kafka_streaming.monitor"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")


@component(base_image="hirschazer/producer:latest")
def producer_op(output_metadata: OutputPath(str)):
    import subprocess
    """Phase-3 Kafka 生产者"""
    subprocess.run(["python", "-m", "kafka_streaming.producer"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")


@component(base_image="hirschazer/infer:latest")
def inference_op(output_metadata: OutputPath(str)):
    import subprocess
    """Phase-4 在线推理 Consumer"""
    subprocess.run(["python", "-m", "kafka_streaming.inference_consumer"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")


@component(base_image="hirschazer/plot:latest")
def plot_op(output_metadata: OutputPath(str)):
    import subprocess
    """Phase-5 最终绘图"""
    subprocess.run(["python", "-m", "kafka_streaming.plot_final"], check=True)
    with open(output_metadata, "w") as f:
        f.write("{}")


# ────────────────────────────────────────────────────────────────
# 2. 组装 Pipeline 拓扑
# ────────────────────────────────────────────────────────────────

@pipeline(name="drift-demo-k8s-stream-pytorch")
def drift_stream():
    off = offline_op().set_display_name("offline-training").set_caching_options(False)
    mon = monitor_op().after(off).set_display_name("drift-monitor").set_caching_options(False)
    prod = producer_op().after(off).set_display_name("kafka-producer").set_caching_options(False)

    consumers = []
    for i in range(3):
        c = inference_op().after(off).set_display_name(f"online-inference-{i}").set_caching_options(False)
        consumers.append(c)

    plot_op().after(prod, *consumers).set_display_name("draw-graph-op").set_caching_options(False)


# ────────────────────────────────────────────────────────────────
# 3. 本地编译 YAML
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v6.yaml",
    )
    print("✅ drift_demo_v6.yaml 生成完毕")
