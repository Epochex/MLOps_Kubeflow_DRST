# pipeline/pipeline.py
import json
from kfp.dsl import pipeline, container_component, ContainerSpec
from kfp import compiler

PVC_NAME   = "mlops-custom3-workspace"
MOUNT_PATH = "/mnt/pvc"

def make_op(name: str, image: str, cmd: list[str]):
    @container_component
    def _op() -> ContainerSpec:
        # ContainerSpec 里只放镜像 + 命令
        return ContainerSpec(image=image, command=cmd)
    _op.__name__ = name.replace("-", "_")
    return _op

# # 六个组件定义如常
# cleanup  = make_op(
#     "cleanup", "bash:5.1",
#     ["bash","-c",
#      "rm -rf /mnt/pvc/results/*.flag && "
#      "rm -rf /mnt/pvc/results/*        && "
#      "rm -rf /mnt/pvc/models/*         && "
#      "rm -rf /mnt/pvc/datasets_old/*"
#     ]
# )

offline    = make_op("offline",   "hirschazer/offline:latest",
                    ["python","-m","ml.train_offline"])
monitor    = make_op("monitor",   "hirschazer/monitor:latest",
                    ["python","-m","kafka_streaming.monitor"])
producer   = make_op("producer",  "hirschazer/producer:latest",
                    ["python","-m","kafka_streaming.producer"])
inference  = make_op("inference", "hirschazer/infer:latest",
                    ["python","-m","kafka_streaming.inference_consumer"])
plot       = make_op("plot",      "hirschazer/plot:latest",
                    ["python","-m","kafka_streaming.plot_final"])

def pvc(task):
    """
    给每个 Task 打上一模一样的 PVC patch。
    注意：要给 task.pod_spec_patch 赋值一个 JSON 字符串，
    而且最顶层要有 "spec":{ volumes, containers }。
    """
    patch = {
        "spec": {
            "volumes": [
                {
                    "name": PVC_NAME,
                    "persistentVolumeClaim": {
                        "claimName": PVC_NAME
                    }
                }
            ],
            "containers": [
                {
                    "name": "driver",           # v2 执行容器统统叫 driver
                    "volumeMounts": [
                        {"name": PVC_NAME, "mountPath": MOUNT_PATH}
                    ]
                }
            ]
        }
    }
    # **一定** 要 json.dumps，而不是直接给 dict
    task.pod_spec_patch = json.dumps(patch)
    return task

@pipeline(name="drift-demo-k8s-stream-pytorch")
def drift_stream():
    # # 0) 先清理残留（所有后续都 after 这个）
    # clean = pvc(
    #     cleanup()
    #       .set_display_name("cleanup-pvc")
    #       .set_caching_options(False)
    # )

    # 1) 离线训练
    off = pvc(
        offline()
          # .after(clean)
          .set_display_name("offline-training")
          .set_caching_options(False)
    )

    # 2) 并行起 Monitor + Producer
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

    # 3) N 个 Inference
    consumers = []
    for i in range(3):
        c = pvc(
            inference()
              .after(off)
              .set_display_name(f"online-inference-{i}")
              .set_caching_options(False)
        )
        consumers.append(c)

    # 4) 最后画图 （等待 prod + consumers）
    pvc(
        plot()
          .after(prod, *consumers)
          .set_display_name("draw-graph-op")
          .set_caching_options(False)
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v6.yaml"
    )
    print("✅ drift_demo_v6.yaml 生成完毕")
