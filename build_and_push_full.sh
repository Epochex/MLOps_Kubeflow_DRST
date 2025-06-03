#!/usr/bin/env bash
# build_and_push.sh  –  一键构建并推送全部组件镜像
set -euo pipefail
set -x

# ⇣ 你的 Docker Hub / Harbor 用户名
REPO=hirschazer

build_push() {   # $1 = target 名,  $2 = tag(同 target 名即可)
  docker build --target "$1" -t "${REPO}/$2:latest" .
  docker push  "${REPO}/$2:latest"
}

# ──────────────────────────────────────────────────────────────
build_push producer        producer
build_push offline         offline
build_push infer           infer
build_push monitor         monitor
build_push plot            plot

# 保留的离线工具
build_push divergence      divergence
build_push compute-mapping compute-mapping

# 三档重训练
build_push stage1          stage1
build_push stage2          stage2
build_push stage3          stage3
