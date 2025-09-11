#! /bin/bash
set -euxo pipefail  

bash docker/build_and_push.sh
python3 experiments/kubeflow/clean_minio.py
python3 experiments/kubeflow/submit_pipeline.py