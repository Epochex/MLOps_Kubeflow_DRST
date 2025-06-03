#!/usr/bin/env bash
# 构建并推送所有组件镜像
set -euo pipefail
set -x

REPO=hirschazer     # ⇠ 你的仓库

build_push() {
  local TARGET=$1
  docker build --target "$TARGET" -t "${REPO}/${TARGET}:latest" .
  docker push  "${REPO}/${TARGET}:latest"
}

build_push offline
build_push producer
build_push monitor
build_push infer
# build_push stage1
# build_push stage2
# build_push stage3
build_push plot
