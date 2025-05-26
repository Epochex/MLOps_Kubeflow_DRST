########################
# 0. 基础镜像 & 系统依赖
########################
FROM python:3.11-slim AS base
WORKDIR /app

ENV MINIO_ENDPOINT="45.149.207.13:9000" \
    MINIO_ACCESS_KEY="minio" \
    MINIO_SECRET_KEY="minio123" \
    MINIO_BUCKET="onvm-demo1" \
    PYTHONPATH="/app"

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

########################
# 1. 安装 PyTorch CPU 轮子
########################
# 用官方索引安装 CPU 版 torch
RUN pip install --no-cache-dir \
      torch==2.3.1+cpu \
      torchvision==0.18.1+cpu \
      torchaudio==2.3.1+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html

########################
# 2. 安装其余 Python 依赖
########################
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U -r requirements.txt

########################
# 3. 拷贝项目源码（变动层）
########################
COPY . /app

########################
# 4. 三个组件镜像
########################
FROM base AS producer
ENTRYPOINT ["python", "-m", "kafka_streaming.producer"]

FROM base AS offline
ENTRYPOINT ["python", "-m", "ml.train_offline"]

FROM base AS consumer
ENTRYPOINT ["python", "-m", "kafka_streaming.consumer"]

