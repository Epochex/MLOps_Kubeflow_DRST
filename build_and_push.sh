#!/usr/bin/env bash
set -euo pipefail
set -x

# 你自己的 Docker Hub 或私有仓库前缀
REPO=hirschazer

# 构建并推送 Producer 镜像
docker build --target producer -t ${REPO}/producer:latest .
docker push ${REPO}/producer:latest

# 构建并推送 Offline 镜像
docker build --target offline  -t ${REPO}/offline:latest  .
docker push ${REPO}/offline:latest

# 构建并推送 Consumer 镜像
docker build --target consumer -t ${REPO}/consumer:latest .
docker push ${REPO}/consumer:latest

