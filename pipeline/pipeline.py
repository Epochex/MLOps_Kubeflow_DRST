from kfp.dsl import pipeline, container_component, ContainerSpec
from kfp import compiler

PVC_NAME, MOUNT_PATH = "data-pvc", "/mnt/pvc"

def make_op(name, image, cmd):
    @container_component
    def _op() -> ContainerSpec:
        return ContainerSpec(image=image, command=cmd)
    _op.__name__ = name.replace("-", "_")
    return _op

offline   = make_op("offline",   "hirschazer/offline:latest",
                    ["python", "-m", "ml.train_offline"])
producer  = make_op("producer",  "hirschazer/producer:latest",
                    ["python", "-m", "kafka_streaming.producer"])
monitor   = make_op("monitor",   "hirschazer/monitor:latest",
                    ["python", "-m", "kafka_streaming.monitor"])
inference = make_op("inference", "hirschazer/infer:latest",
                    ["python", "-m", "kafka_streaming.inference_consumer"])
plot      = make_op("plot",      "hirschazer/plot:latest",
                    ["python", "-m", "kafka_streaming.plot_final"])

def pvc(task):
    task.pod_spec_patch = {
        "volumes": [{
            "name": "data-vol",
            "persistentVolumeClaim": {"claimName": PVC_NAME}
        }],
        "containers": [{
            "name": "main",
            "volumeMounts": [{"mountPath": MOUNT_PATH, "name": "data-vol"}]
        }],
    }
    return task

@pipeline(name="drift-demo-k8s-stream-pytorch")
def drift_stream():
    off = pvc(
        offline().set_display_name("offline-training").set_caching_options(False)
    )

    # ↓↓↓ 三个组件 **并行** ↓↓↓
    mon  = pvc(monitor().after(off).set_display_name("monitor")
                                  .set_caching_options(False))
    inf  = pvc(inference().after(off).set_display_name("online-inference")
                                     .set_caching_options(False))
    prod = producer().after(off).set_display_name("producer")\
                             .set_caching_options(False)

    pvc(plot().after(prod, inf)
              .set_display_name("draw-graph-op")
              .set_caching_options(False))

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=drift_stream,
        package_path="drift_demo_v8.yaml"
    )
    print("✅ 生成 drift_demo_v8.yaml")
