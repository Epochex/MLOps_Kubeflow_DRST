# pipeline/pipeline.py
from kfp.dsl import pipeline, container_component, ContainerSpec
from kfp import compiler

PVC_NAME, MOUNT_PATH = "mlops-custom3-workspace", "/mnt/pvc"

def make_op(name, image, cmd):
    @container_component
    def _op() -> ContainerSpec:
        return ContainerSpec(image=image, command=cmd)
    _op.__name__ = name.replace("-", "_")
    return _op

offline    = make_op("offline",   "hirschazer/offline:latest",
                    ["python", "-m", "ml.train_offline"])
producer   = make_op("producer",  "hirschazer/producer:latest",
                    ["python", "-m", "kafka_streaming.producer"])
monitor    = make_op("monitor",   "hirschazer/monitor:latest",
                    ["python", "-m", "kafka_streaming.monitor"])
inference  = make_op("inference", "hirschazer/infer:latest",
                    ["python", "-m", "kafka_streaming.inference_consumer"])
plot       = make_op("plot",      "hirschazer/plot:latest",
                    ["python", "-m", "kafka_streaming.plot_final"])

def pvc(task):
    task.pod_spec_patch = {
        "volumes": [{
            "name": PVC_NAME,
            "persistentVolumeClaim": {"claimName": PVC_NAME}
        }],
        "containers": [{
            "name": "main",
            "volumeMounts": [{"mountPath": MOUNT_PATH, "name": PVC_NAME}]
        }],
    }
    return task

@pipeline(name="drift-demo-k8s-stream-pytorch")
def drift_stream():
    # 1) 离线训好模型
    off = pvc(
        offline()
          .set_display_name("offline-training")
          .set_caching_options(False)
    )

    # 2) 并行启动 Monitor + Producer
    mon  = pvc(
        monitor()
          .after(off)
          .set_display_name("drift-monitor")
          .set_caching_options(False)
    )
    prod = pvc(
        producer()
          .after(off)
          .set_display_name("kafka-producer")
          .set_caching_options(False)
    )

    # 3) 水平扩展 N 个 Inference Consumers（N = 分区数 = 3）
    N = 3
    consumer_ops = []
    for i in range(N):
        consumer = pvc(
            inference()
              .after(off)
              .set_display_name(f"online-inference-{i}")
              .set_caching_options(False)
        )
        consumer_ops.append(consumer)

    # 4) 最后等 Producer + 全部 Consumers 都结束，再画图
    #    .after() 可以接受多个任务
    plot_deps = [prod] + consumer_ops
    pvc(
        plot()
          .after(*plot_deps)
          .set_display_name("draw-graph-op")
          .set_caching_options(False)
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v6.yaml"
    )
    print("✅ 生成 drift_demo_v6.yaml")
