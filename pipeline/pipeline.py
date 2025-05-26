# pipeline.py

from kfp import dsl
from kfp.dsl import container_component, ContainerSpec

@container_component
def offline_train_op() -> ContainerSpec:
    return ContainerSpec(
        image="hirschazer/offline:latest",
        command=["python", "-m", "ml.train_offline"],
    )

@container_component
def producer_op() -> ContainerSpec:
    return ContainerSpec(
        image="hirschazer/producer:latest",
        command=["python", "-m", "kafka_streaming.producer"],
    )

@container_component
def consumer_op() -> ContainerSpec:
    return ContainerSpec(
        image="hirschazer/consumer:latest",
        command=["python", "-m", "kafka_streaming.consumer"],
    )

@container_component
def plot_final_op() -> ContainerSpec:
    return ContainerSpec(
        image="hirschazer/consumer:latest",
        command=["python", "-m", "kafka_streaming.plot_final"],
    )

@dsl.pipeline(name="drift-detect-demo-v2")
def drift_demo_pipeline():
    # 1) 离线训练（模型 & PCA 等）
    offline = offline_train_op()

    # 2) 生产者 & 消费者 并行运行
    prod = producer_op().after(offline)
    cons = consumer_op().after(offline)

    # 3) 最终绘图 依赖 两者完成
    _ = plot_final_op().after(prod, cons)
