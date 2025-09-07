#! /bin/bash
set -euxo pipefail   # <- 多一个 -x 便于排错

bash docker/build_and_push.sh
python3 experiments/kubeflow/clean_minio.py
python3 experiments/kubeflow/submit_pipeline.py
