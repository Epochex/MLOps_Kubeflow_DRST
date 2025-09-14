# DRST-SoftwarizedNetworks/experiments/kubeflow/pipeline.py
# -*- coding: utf-8 -*-
"""
Kubeflow Pipeline (v2)
- PCM-Pre-op：运行 drst_preprocess.pcm.preprocess_pcm 与 drst_preprocess.pcm.extract_pcm
- Perf-Pre-op：运行 drst_preprocess.perf.preprocess_perf 与 drst_preprocess.perf.extract_perf

依赖关系：
- Perf-Pre-op  →  Offline-Training-op
- PCM-Pre-op   →  Forecast-GridSearch-op → Forecast-XAI-op
"""

import os
from kfp import dsl
from kfp.dsl import component, pipeline

# 镜像（可通过环境变量在编译时替换；留空就用仓库里默认值）
IMG_OFFLINE:   str = os.getenv("IMAGE_OFFLINE",   "hirschazer/offline:latest")
IMG_MONITOR:   str = os.getenv("IMAGE_MONITOR",   "hirschazer/monitor:latest")
IMG_PRODUCER:  str = os.getenv("IMAGE_PRODUCER",  "hirschazer/producer:latest")
IMG_INFER:     str = os.getenv("IMAGE_INFER",     "hirschazer/infer:latest")
IMG_PLOT:      str = os.getenv("IMAGE_PLOT",      "hirschazer/plot:latest")
IMG_RETRAIN:   str = os.getenv("IMAGE_RETRAIN",   "hirschazer/retrain:latest")
IMG_FORECAST:  str = os.getenv("IMAGE_FORECAST",  "hirschazer/forecast:latest")

# 复用 offline 镜像来跑预处理（镜像里已包含 /app/drst_preprocess）
IMG_PREPROC:   str = os.getenv("IMAGE_PREPROC",   IMG_OFFLINE)

# ========== （如需要PVC可保留；若仅走MinIO，挂载可不生效也不影响） ==========
DATA_PVC_NAME: str = os.getenv("DATA_PVC_NAME", "YOUR_DATA_PVC")
DATA_MOUNT     = "/data"

# ---------------------------- 预处理组件 ----------------------------
@dsl.container_component
def pcm_pre_op():
    return dsl.ContainerSpec(
        image=IMG_PREPROC,
        command=["bash","-lc"],
        args=[r"""
            set -e
            export PYTHONPATH=/app
            echo "[PCM-Pre-op] preprocess -> extract (MinIO I/O inside scripts)"
            /opt/venv/bin/python -V
            which python || true
            /opt/venv/bin/python -c "import sys; print(sys.executable)"
            /opt/venv/bin/python -c "import pandas, numpy; print('pandas', pandas.__version__, 'numpy', numpy.__version__)"
            /opt/venv/bin/python -m drst_preprocess.pcm.preprocess_pcm
            /opt/venv/bin/python -m drst_preprocess.pcm.extract_pcm
            echo "[PCM-Pre-op] done."
        """],
    )

@dsl.container_component
def perf_pre_op():
    return dsl.ContainerSpec(
        image=IMG_PREPROC,
        command=["bash","-lc"],
        args=[r"""
            set -e
            export PYTHONPATH=/app
            echo "[Perf-Pre-op] preprocess -> extract (MinIO I/O inside scripts)"
            /opt/venv/bin/python -V
            which python || true
            /opt/venv/bin/python -c "import sys; print(sys.executable)"
            /opt/venv/bin/python -c "import pandas, numpy; print('pandas', pandas.__version__, 'numpy', numpy.__version__)"
            /opt/venv/bin/python -m drst_preprocess.perf.preprocess_perf
            /opt/venv/bin/python -m drst_preprocess.perf.extract_perf
            echo "[Perf-Pre-op] done."
        """],
    )


# ---------------------------- 其余组件（与你现有保持一致） ----------------------------
@component(base_image=IMG_OFFLINE)
def offline_training_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    subprocess.run(["python", "-m", "drst_inference.offline.train_offline"], check=True)

@component(base_image=IMG_PRODUCER)
def producer_op(
    kafka_topic: str = "latencyTopic",
    interval_ms: int = 200,
    producer_stages: str = "",
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if kafka_topic:
        os.environ["KAFKA_TOPIC"] = str(kafka_topic)
    os.environ["INTERVAL_MS"] = str(int(interval_ms))
    if producer_stages:
        os.environ["PRODUCER_STAGES"] = producer_stages
    subprocess.run(["python", "-m", "drst_inference.online.producer"], check=True)

@component(base_image=IMG_MONITOR)
def monitor_op(max_wall_secs: int = 0) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if int(max_wall_secs) > 0:
        os.environ["MAX_WALL_SECS"] = str(int(max_wall_secs))
    subprocess.run(["python", "-m", "drst_drift.monitor"], check=True)

@component(base_image=IMG_RETRAIN)
def retrain_op(watch: bool = True, poll_interval_s: int = 2) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["RETRAIN_WATCH"]   = "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"] = str(int(poll_interval_s))
    subprocess.run(["python", "-m", "drst_drift.dynamic_retrain"], check=True)

@component(base_image=IMG_INFER)
def infer_op(replica_id: int, kafka_topic: str = "latencyTopic") -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if kafka_topic:
        os.environ["KAFKA_TOPIC"] = str(kafka_topic)
    os.environ["INFER_REPLICA_ID"] = str(int(replica_id))
    subprocess.run(["python", "-m", "drst_inference.online.inference_consumer"], check=True)

@component(base_image=IMG_PLOT)
def plot_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
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

@component(base_image=IMG_FORECAST)
def forecast_gridsearch_op(
    lookback: int = 10,
    horizon: int = 5,
    epochs: int = 50,
    patience: int = 8,
    hidden: int = 64,
    layers: int = 1,
    batch_size: int = 64,
    take_last: int = 0,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["FORECAST_LOOKBACK"]  = str(int(lookback))
    os.environ["FORECAST_HORIZON"]   = str(int(horizon))
    os.environ["FORECAST_EPOCHS"]    = str(int(epochs))
    os.environ["FORECAST_PATIENCE"]  = str(int(patience))
    os.environ["FORECAST_HIDDEN"]    = str(int(hidden))
    os.environ["FORECAST_LAYERS"]    = str(int(layers))
    os.environ["FORECAST_BS"]        = str(int(batch_size))
    os.environ["FORECAST_TAKE_LAST"] = str(int(take_last))
    subprocess.run(["python", "-m", "drst_forecasting.train_benchmark"], check=True)

@component(base_image=IMG_FORECAST)
def forecast_xai_op(
    lookback: int = 10,
    horizon: int = 5,
    shap_n: int = 256,
    hidden: int = 64,
    layers: int = 1,
) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["FORECAST_LOOKBACK"] = str(int(lookback))
    os.environ["FORECAST_HORIZON"]  = str(int(horizon))
    os.environ["FORECAST_SHAP_N"]   = str(int(shap_n))
    os.environ["FORECAST_HIDDEN"]   = str(int(hidden))
    os.environ["FORECAST_LAYERS"]   = str(int(layers))
    subprocess.run(["python", "-m", "drst_forecasting.explain"], check=True)

# ---------------------------- Pipeline ----------------------------
@pipeline(
    name="drift-stream-v2",
    description="Drift monitoring + dynamic retraining + online inference + preprocessing (PCM/Perf)"
)
def drift_stream_v2_pipeline(
    kafka_topic: str = "latencyTopic",
    producer_interval_ms: int = 200,
    producer_stages: str = "",
    monitor_max_wall_secs: int = 0,
    fc_lookback: int = 10,
    fc_horizon:  int = 5,
) -> None:

    # 0) 预处理（可并行）
    pcm = pcm_pre_op().set_caching_options(False)
    pcm.set_display_name("PCM-Pre-op")

    perf = perf_pre_op().set_caching_options(False)
    perf.set_display_name("Perf-Pre-op")

    # 1) offline 仅依赖 Perf（使用 datasets/combined.csv + perf 产物）
    offline = offline_training_op().set_caching_options(False)
    offline.set_display_name("Offline-Training-op")
    offline.after(perf)

    # 2) producer / monitor / retrain / infer 都依赖 offline
    producer = producer_op(
        kafka_topic=kafka_topic,
        interval_ms=producer_interval_ms,
        producer_stages=producer_stages,
    ).set_caching_options(False)
    producer.after(offline)
    producer.set_display_name("Producer-op")

    monitor = monitor_op(max_wall_secs=monitor_max_wall_secs).set_caching_options(False)
    monitor.after(offline)
    monitor.set_display_name("Monitor-op")

    retrain = retrain_op(watch=True, poll_interval_s=2).set_caching_options(False)
    retrain.after(offline)
    retrain.set_display_name("Retrain-op")

    infer0 = infer_op(replica_id=0, kafka_topic=kafka_topic).set_caching_options(False)
    infer1 = infer_op(replica_id=1, kafka_topic=kafka_topic).set_caching_options(False)
    infer2 = infer_op(replica_id=2, kafka_topic=kafka_topic).set_caching_options(False)
    infer0.after(offline); infer1.after(offline); infer2.after(offline)
    infer0.set_display_name("Infer-1-op")
    infer1.set_display_name("Infer-2-op")
    infer2.set_display_name("Infer-3-op")

    # 3) forecasting：仅依赖 PCM 预处理
    fc_grid = forecast_gridsearch_op(
        lookback=fc_lookback, horizon=fc_horizon
    ).set_caching_options(False)
    fc_grid.after(pcm)
    fc_grid.set_display_name("Forecast-GridSearch-op")

    fc_xai = forecast_xai_op(
        lookback=fc_lookback, horizon=fc_horizon
    ).set_caching_options(False)
    fc_xai.after(fc_grid)
    fc_xai.set_display_name("Forecast-XAI-op")

    # 4) plot 汇总（在所有可结束节点之后）
    plot = plot_op().set_caching_options(False)
    plot.after(producer, monitor, retrain, infer0, infer1, infer2, fc_xai)
    plot.set_display_name("Plot-Report-op")


if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=drift_stream_v2_pipeline,
        package_path="drift_stream_v2.json"
    )
