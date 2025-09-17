# DRST-SoftwarizedNetworks/experiments/kubeflow/pipeline.py
# -*- coding: utf-8 -*-

import os
from kfp import dsl
from kfp.dsl import component, pipeline

IMG_OFFLINE  = os.getenv("IMAGE_OFFLINE",  "hirschazer/offline:latest")
IMG_MONITOR  = os.getenv("IMAGE_MONITOR",  "hirschazer/monitor:latest")
IMG_PRODUCER = os.getenv("IMAGE_PRODUCER", "hirschazer/producer:latest")
IMG_INFER    = os.getenv("IMAGE_INFER",    "hirschazer/infer:latest")
IMG_PLOT     = os.getenv("IMAGE_PLOT",     "hirschazer/plot:latest")
IMG_RETRAIN  = os.getenv("IMAGE_RETRAIN",  "hirschazer/retrain:latest")
IMG_FORECAST = os.getenv("IMAGE_FORECAST", "hirschazer/forecast:latest")
IMG_HPSEARCH = os.getenv("IMAGE_HPSEARCH", IMG_FORECAST)
IMG_PREPROC  = os.getenv("IMAGE_PREPROC",  IMG_OFFLINE)

# ------------ 预处理 ------------
@dsl.container_component
def pcm_pre_op():
    return dsl.ContainerSpec(
        image=IMG_PREPROC,
        command=["bash","-lc"],
        args=[r"""
            set -e
            export PYTHONPATH=/app
            /opt/venv/bin/python -m drst_preprocess.pcm.preprocess_pcm
            /opt/venv/bin/python -m drst_preprocess.pcm.extract_pcm
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
            /opt/venv/bin/python -m drst_preprocess.perf.preprocess_perf
            /opt/venv/bin/python -m drst_preprocess.perf.extract_perf
        """],
    )

# ------------ Offline / Online / Plot / Retrain ------------
@component(base_image=IMG_OFFLINE)
def offline_training_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    subprocess.run(["python","-m","drst_inference.offline.train_offline"], check=True)

@component(base_image=IMG_PRODUCER)
def producer_op(kafka_topic: str = "latencyTopic", interval_ms: int = 200, producer_stages: str = "") -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if kafka_topic: os.environ["KAFKA_TOPIC"] = kafka_topic
    os.environ["INTERVAL_MS"] = str(int(interval_ms))
    if producer_stages: os.environ["PRODUCER_STAGES"] = producer_stages
    subprocess.run(["python","-m","drst_inference.online.producer"], check=True)

@component(base_image=IMG_MONITOR)
def monitor_op(max_wall_secs: int = 0) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if int(max_wall_secs) > 0: os.environ["MAX_WALL_SECS"] = str(int(max_wall_secs))
    subprocess.run(["python","-m","drst_drift.monitor"], check=True)

@component(base_image=IMG_RETRAIN)
def retrain_op(watch: bool = True, poll_interval_s: int = 2) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["RETRAIN_WATCH"] = "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"] = str(int(poll_interval_s))
    subprocess.run(["python","-m","drst_drift.dynamic_retrain"], check=True)

@component(base_image=IMG_INFER)
def infer_op(replica_id: int, kafka_topic: str = "latencyTopic") -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["INFER_REPLICA_ID"] = str(int(replica_id))
    if kafka_topic: os.environ["KAFKA_TOPIC"] = kafka_topic
    subprocess.run(["python","-m","drst_inference.online.inference_consumer"], check=True)

@component(base_image=IMG_PLOT)
def plot_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    for cmd in [
        ["python","-m","drst_inference.plotting.plot_final"],
        ["python","experiments/kubeflow/summarize_resources.py"],
        ["python","-m","drst_inference.plotting.plot_report"],
    ]:
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[plot_op] skipped {cmd}: {e}", flush=True)

# ------------ Model Selection ------------
@component(base_image=IMG_FORECAST)
def pcm_model_selection_op(lookback: int = 10, horizon: int = 5, take_last: int = 4000, topk: int = 3) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    subprocess.run([
        "python","-m","drst_model_selection.cli","--task=pcm",
        f"--lookback={int(lookback)}", f"--horizon={int(horizon)}",
        f"--take_last={int(take_last)}", f"--topk={int(topk)}",
    ], check=True)

@component(base_image=IMG_FORECAST)
def perf_model_selection_op(perf_key: str = "datasets/perf/stage1_random_rates.csv", topk: int = 4, include_svr: int = 0, include_dt: int = 0) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    subprocess.run([
        "python","-m","drst_model_selection.cli","--task=perf",
        f"--perf_key={perf_key}", f"--topk={int(topk)}",
        f"--include_svr={int(include_svr)}", f"--include_dt={int(include_dt)}",
    ], check=True)

# ------------ HP Search（PCM/Perf）------------
@component(base_image=IMG_HPSEARCH)
def perf_hpsearch_op(data_path: str = "datasets/combined.csv", n_jobs: int = 0, torch_threads: int = 1) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if int(n_jobs) > 0: os.environ["N_JOBS"] = str(int(n_jobs))
    os.environ["TORCH_NUM_THREADS"] = str(int(torch_threads))
    cmd = ["python","-m","drst_hpsearch.perf_hpsearch", f"--data_path={data_path}"]
    if int(n_jobs) > 0: cmd.append(f"--n_jobs={int(n_jobs)}")
    subprocess.run(cmd, check=True)

@component(base_image=IMG_HPSEARCH)
def pcm_hpsearch_op(lookback: int = 10, horizon: int = 5, take_last: int = 4000, n_jobs: int = 0, torch_threads: int = 1) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    if int(n_jobs) > 0: os.environ["N_JOBS"] = str(int(n_jobs))
    os.environ["TORCH_NUM_THREADS"] = str(int(torch_threads))
    os.environ["FORECAST_LOOKBACK"] = str(int(lookback))
    os.environ["FORECAST_HORIZON"]  = str(int(horizon))
    os.environ["FORECAST_TAKE_LAST"]= str(int(take_last))
    subprocess.run(["python","-m","drst_hpsearch.pcm_hpsearch"], check=True)

# ------------ Forecast（发布，非搜索）------------
@component(base_image=IMG_FORECAST)
def forecast_publish_op() -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    subprocess.run(["python","-m","drst_forecasting.publish_forecast"], check=True)

# ------------ XAI（跟随 retrain / latest 指针）------------
@component(base_image=IMG_FORECAST)
def forecast_xai_op(lookback: int = 10, horizon: int = 5, shap_n: int = 256, hidden: int = 64, layers: int = 1, watch: bool = True, poll_interval_s: int = 2) -> None:
    import os, subprocess
    os.environ["PYTHONPATH"] = "/app"
    os.environ["FORECAST_LOOKBACK"] = str(int(lookback))
    os.environ["FORECAST_HORIZON"]  = str(int(horizon))
    os.environ["FORECAST_SHAP_N"]   = str(int(shap_n))
    os.environ["FORECAST_HIDDEN"]   = str(int(hidden))
    os.environ["FORECAST_LAYERS"]   = str(int(layers))
    os.environ["FORECAST_XAI_WATCH"]= "1" if watch else "0"
    os.environ["POLL_INTERVAL_S"]   = str(int(poll_interval_s))
    subprocess.run(["python","-m","drst_forecasting.explain"], check=True)

# ------------ Pipeline ------------
@pipeline(name="drift-stream-v2", description="Drift monitoring + dynamic retraining + online inference + PCM/Perf preprocess + HP search + Forecast publish")
def drift_stream_v2_pipeline(
    kafka_topic: str = "latencyTopic",
    producer_interval_ms: int = 200,
    producer_stages: str = "",
    monitor_max_wall_secs: int = 0,
    fc_lookback: int = 10,
    fc_horizon: int = 5,
    fc_take_last: int = 4000,
    hp_n_jobs: int = 0,
    hp_torch_threads: int = 1,
) -> None:

    # 预处理
    pcm = pcm_pre_op().set_caching_options(False).set_display_name("PCM-Pre-op")
    perf = perf_pre_op().set_caching_options(False).set_display_name("Perf-Pre-op")

    # Model selection
    perf_sel = perf_model_selection_op(
        perf_key="datasets/perf/stage1_random_rates.csv",
        topk=4, include_svr=0, include_dt=0
    ).set_caching_options(False).set_display_name("Perf-ModelSelection-op").after(perf)

    pcm_sel = pcm_model_selection_op(
        lookback=fc_lookback, horizon=fc_horizon, take_last=fc_take_last, topk=3
    ).set_caching_options(False).set_display_name("PCM-ModelSelection-op").after(pcm)

    # HP search
    perf_hp = perf_hpsearch_op(
        data_path="datasets/combined.csv",
        n_jobs=hp_n_jobs, torch_threads=hp_torch_threads
    ).set_caching_options(False).set_display_name("Perf-HPsearch-op").after(perf_sel)

    pcm_hp = pcm_hpsearch_op(
        lookback=fc_lookback, horizon=fc_horizon, take_last=fc_take_last,
        n_jobs=hp_n_jobs, torch_threads=hp_torch_threads
    ).set_caching_options(False).set_display_name("PCM-HPsearch-op").after(pcm_sel)

    # Offline → Producer/Monitor/Retrain/Infer
    offline = offline_training_op().set_caching_options(False).set_display_name("Offline-Training-op").after(perf_hp)

    producer = producer_op(kafka_topic=kafka_topic, interval_ms=producer_interval_ms, producer_stages=producer_stages
    ).set_caching_options(False).set_display_name("Producer-op").after(offline)

    monitor = monitor_op(max_wall_secs=monitor_max_wall_secs
    ).set_caching_options(False).set_display_name("Monitor-op").after(offline)

    retrain = retrain_op(watch=True, poll_interval_s=2
    ).set_caching_options(False).set_display_name("Retrain-op").after(offline)

    infer0 = infer_op(replica_id=0, kafka_topic=kafka_topic).set_caching_options(False).set_display_name("Infer-1-op").after(offline)
    infer1 = infer_op(replica_id=1, kafka_topic=kafka_topic).set_caching_options(False).set_display_name("Infer-2-op").after(offline)
    infer2 = infer_op(replica_id=2, kafka_topic=kafka_topic).set_caching_options(False).set_display_name("Infer-3-op").after(offline)

    # Forecast 发布：仅依赖 PCM-HPsearch + Offline（确保模型已选好、基础线起完）
    fc_pub = forecast_publish_op().set_caching_options(False).set_display_name("Forecast-Publish-op").after(pcm_hp, offline)

    # XAI：依赖 offline（内部 watch retrain/latest）
    fc_xai = forecast_xai_op(lookback=fc_lookback, horizon=fc_horizon, watch=True, poll_interval_s=2
    ).set_caching_options(False).set_display_name("Forecast-XAI-op").after(offline)

    # Plot 收口
    plot = plot_op().set_caching_options(False).set_display_name("Plot-Report-op")
    plot.after(producer, monitor, retrain, infer0, infer1, infer2, fc_pub, fc_xai)

if __name__ == "__main__":
    import kfp
    kfp.compiler.Compiler().compile(
        pipeline_func=drift_stream_v2_pipeline,
        package_path="drift_stream_v2.json"
    )
