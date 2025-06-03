# ──────────────────────────────────────────────────────────────
# Dockerfile  (multi-stage，覆盖 offline / producer / monitor /
#              infer / plot 五个 Target，与 Kubeflow Pipeline 相配)
# ──────────────────────────────────────────────────────────────

############################  common base  ############################
FROM python:3.11-slim AS base
WORKDIR /app

# —— 通用环境变量 —— ---------------------------------------------------
ENV PYTHONPATH=/app \
    # MinIO
    MINIO_ENDPOINT=45.149.207.13:9000 \
    MINIO_ACCESS_KEY=minio \
    MINIO_SECRET_KEY=minio123 \
    MINIO_BUCKET=onvm-demo2 \
    # 结果 / PVC 挂载点
    RESULT_DIR=results

# —— 系统依赖（最小化） —— ---------------------------------------------
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        build-essential git \
 && rm -rf /var/lib/apt/lists/*

# —— 先复制 requirements.txt，再安装依赖 —— ---------------------------
COPY requirements.txt /app/


RUN pip install --no-cache-dir \
        torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html  \
 && pip install --no-cache-dir -U -r requirements.txt          \
 && pip install --no-cache-dir psutil                          \
 && pip cache purge

# —— 复制全部项目代码（包括 shared.metric_logger 等新文件） ————————
COPY . /app

############################  component targets  ############################

# 1. 离线基线训练（train_offline）
FROM base AS offline
ENTRYPOINT ["python", "-m", "ml.train_offline"]

# 2. 数据生产者（Kafka Producer）
FROM base AS producer
ENTRYPOINT ["python", "-m", "kafka_streaming.producer"]

# 3. Drift 监控 + 动态重训触发器
FROM base AS monitor
ENTRYPOINT ["python", "-m", "kafka_streaming.monitor"]

# 4. 在线推理 Consumer（带性能指标采集）
FROM base AS infer
ENTRYPOINT ["python", "-m", "kafka_streaming.inference_consumer"]

# 5. 结果绘图组件
FROM base AS plot
ENTRYPOINT ["python", "-m", "kafka_streaming.plot_final"]

# ──────────────────────────────────────────────────────────────
# 用法示例（一次 build 多次推镜像）：
#
#   docker build --target offline  -t <repo>/offline:latest  .
#   docker push  <repo>/offline:latest
#
#   docker build --target infer    -t <repo>/infer:latest    .
#   docker push  <repo>/infer:latest
#
# 其余 producer / monitor / plot 同理。
# ──────────────────────────────────────────────────────────────
