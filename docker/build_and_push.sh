# /data/mlops/DRST-SoftwarizedNetworks/docker/build_and_push.sh
#!/usr/bin/env bash
set -euo pipefail

REPO="${DOCKER_REPO:-hirschazer}"
TAG="${TAG:-latest}"
FILE="${DOCKERFILE:-docker/dockerfile}"

# 准备 buildx（若还没有）
docker buildx inspect drst-builder >/dev/null 2>&1 || \
  docker buildx create --use --name drst-builder

# 可选：把缓存推到本地目录，加速多 target 连续构建
CACHE_DIR="${CACHE_DIR:-/tmp/.drst-buildx-cache}"

build_one () {
  local target="$1" ; local image="$2"
  echo "==> building ${image}:${TAG} (target=${target})"
  docker buildx build \
    --file "${FILE}" \
    --target "${target}" \
    --tag   "${image}:${TAG}" \
    --cache-from "type=local,src=${CACHE_DIR}" \
    --cache-to   "type=local,dest=${CACHE_DIR},mode=max" \
    --push \
    .
}

build_one offline  "${REPO}/offline"
build_one monitor  "${REPO}/monitor"
build_one producer "${REPO}/producer"
build_one infer    "${REPO}/infer"
build_one plot     "${REPO}/plot"

echo "######pushed############:"
echo "  ${REPO}/offline:${TAG}"
echo "  ${REPO}/monitor:${TAG}"
echo "  ${REPO}/producer:${TAG}"
echo "  ${REPO}/infer:${TAG}"
echo "  ${REPO}/plot:${TAG}"
