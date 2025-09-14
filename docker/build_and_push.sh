# DRST-SoftwarizedNetworks/docker/build_and_push.sh
#!/usr/bin/env bash
set -euo pipefail

REPO="${DOCKER_REPO:-hirschazer}"
TAG="${TAG:-latest}"
FILE="${DOCKERFILE:-docker/dockerfile}"

docker buildx inspect drst-builder >/dev/null 2>&1 || docker buildx create --use --name drst-builder
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

# 你现有的
build_one offline  "${REPO}/offline"
build_one monitor  "${REPO}/monitor"
build_one producer "${REPO}/producer"
build_one infer    "${REPO}/infer"
build_one plot     "${REPO}/plot"
build_one retrain  "${REPO}/retrain"

# 新增
build_one forecast      "${REPO}/forecast"
build_one forecast_api  "${REPO}/forecast-api"

echo "###### pushed ############:"
for n in offline monitor producer infer plot retrain forecast forecast-api; do
  echo "  ${REPO}/${n}:${TAG}"
done
