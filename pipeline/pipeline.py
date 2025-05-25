import kfp.dsl as dsl

MINIO_SECRET = "minio-secret"

def env_from_secret(var):
    return dsl.EnvVar(name=var, value_from=dsl.EnvVarSource(
        secret_key_ref=dsl.SecretKeySelector(name=MINIO_SECRET, key=var)))

MINIO_ENVS = [env_from_secret(k) for k in
              ("MINIO_ENDPOINT","MINIO_ACCESS_KEY",
               "MINIO_SECRET_KEY","MINIO_BUCKET")]

@dsl.container_component
def offline_train_op():
    return dsl.ContainerSpec(
        image='hirschazer/offline:latest',
        command=['python','-m','ml.train_offline'],
        env=MINIO_ENVS)

@dsl.container_component
def producer_op():
    return dsl.ContainerSpec(
        image='hirschazer/producer:latest',
        command=['python','kafka_streaming/producer.py'],
        env=MINIO_ENVS)

@dsl.container_component
def consumer_op():
    return dsl.ContainerSpec(
        image='hirschazer/consumer:latest',
        command=['python','kafka_streaming/consumer.py'],
        env=MINIO_ENVS)

@dsl.pipeline(name="drift-detect-demo")
def drift_demo_pipeline():
    offline = offline_train_op()
    _prod   = producer_op().after(offline)
    _cons   = consumer_op().after(offline)   # 与 prod 并行

if __name__ == "__main__":
    dsl.Compiler().compile(drift_demo_pipeline, "drift_demo.yaml")
