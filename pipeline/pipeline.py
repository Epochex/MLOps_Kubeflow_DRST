# pipeline/pipeline.py
import json
from kfp.dsl import pipeline, container_component, ContainerSpec
from kfp import compiler

# ---------- 1. 轻量包装：生成 Container Op ----------
def make_op(name: str, image: str, cmd: list[str]):
    """
    返回一个 kfp.dsl.ContainerSpec 包装的组件函数，
    仅包含镜像 + 启动命令。
    """
    @container_component
    def _op() -> ContainerSpec:
        return ContainerSpec(image=image, command=cmd)

    # 让函数名合法且易读
    _op.__name__ = name.replace("-", "_")
    return _op

# ---------- 2. 五大核心组件 ----------
offline   = make_op("offline",   "hirschazer/offline:latest",
                    ["python", "-m", "ml.train_offline"])
monitor   = make_op("monitor",   "hirschazer/monitor:latest",
                    ["python", "-m", "kafka_streaming.monitor"])
producer  = make_op("producer",  "hirschazer/producer:latest",
                    ["python", "-m", "kafka_streaming.producer"])
inference = make_op("inference", "hirschazer/infer:latest",
                    ["python", "-m", "kafka_streaming.inference_consumer"])
plot      = make_op("plot",      "hirschazer/plot:latest",
                    ["python", "-m", "kafka_streaming.plot_final"])

# ---------- 3. Pipeline 拓扑 ----------
@pipeline(name="drift-demo-k8s-stream-pytorch")
def drift_stream():
    # 1) 离线基线训练
    off = (
        offline()
          .set_display_name("offline-training")
          .set_caching_options(False)
    )

    # 2) Monitor & Producer 并行启动（均依赖 offline artefacts）
    mon = (
        monitor()
          .after(off)
          .set_display_name("drift-monitor")
          .set_caching_options(False)
    )
    prod = (
        producer()
          .after(off)
          .set_display_name("kafka-producer")
          .set_caching_options(False)
    )

    # 3) 三个并行 Inference Consumer（同样依赖 offline）
    consumers = []
    for i in range(3):
        c = (
            inference()
              .after(off)
              .set_display_name(f"online-inference-{i}")
              .set_caching_options(False)
        )
        consumers.append(c)

    # 4) 最终绘图，等 Producer + 全部 Consumers 完成
    (
        plot()
          .after(prod, *consumers)
          .set_display_name("draw-graph-op")
          .set_caching_options(False)
    )

# ---------- 4. 本地调试：编译 YAML ----------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v6.yaml"
    )
    print("✅  无 PVC 版本 drift_demo_v6.yaml 生成完毕")