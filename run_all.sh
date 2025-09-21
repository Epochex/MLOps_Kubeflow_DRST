#!/usr/bin/env bash
# One-shot: build images -> upload data -> deploy Online API -> submit/run KFP pipeline
# Fixed server: 45.149.207.13

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NS="kubeflow-user-example-com"
KFP_HOST="http://45.149.207.13:30080/pipeline"

echo "==[1/4] Build & Push=="
bash docker/build_and_push.sh

python3 experiments/kubeflow/clean_minio.py

# echo "==[2/4] Upload datasets to MinIO (gateway: s3.45.149.207.13.nip.io:30080) =="
# python3 deploy/upload_datasets_to_minio.py || echo "[WARN] Upload failed or already exists: ignore if data is already in MinIO"

echo "==[2.1] Build Online API (forecast_api) image =="
docker build -f docker/dockerfile --target forecast_api -t drst/forecast_api:latest .

echo "==[3/4] Deploy Online API (FastAPI / Uvicorn, NodePort: 30081) =="
kubectl -n "$NS" apply -f deploy/services/forecast-api.yaml

echo
echo "==[4/4] Submit and run Kubeflow Pipeline =="
export KFP_HOST
python3 experiments/kubeflow/submit_pipeline.py

cat <<'EOF'

============ Access & Results ============

[Online API (real-time inference service)]
- Docs/self-test (external): http://45.149.207.13:30081/docs
- Health check:              http://45.149.207.13:30081/
- Example call:
  curl -X POST "http://45.149.207.13:30081/predict" \
       -H "Content-Type: application/json" \
       -d '{"inputs":[{"timestamp":"2024-01-01T00:00:00Z","features":{"x1":0.1,"x2":0.2}}]}'

[Kubeflow Pipelines]
- UI:  http://45.149.207.13:30080/pipeline

EOF
