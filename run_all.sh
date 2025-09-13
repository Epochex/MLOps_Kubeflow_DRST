#!/usr/bin/env bash
# 一把梭：构建镜像 -> 上传数据 -> 部署 Online API -> 提交/运行 KFP 流水线
# 服务器固定：45.149.207.13

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NS="kubeflow-user-example-com"
KFP_HOST="http://45.149.207.13:30080/pipeline"

echo "==[1/4] 构建 & 推送（沿用你的脚本）=="
bash docker/build_and_push.sh

python3 experiments/kubeflow/clean_minio.py

# echo "==[2/4] 上传 datasets 到 MinIO（网关：s3.45.149.207.13.nip.io:30080）=="
# python3 deploy/upload_datasets_to_minio.py || echo "[WARN] 上传失败或已存在：若数据已在 MinIO 可忽略"

echo "==[2.1] 构建 Online API（forecast_api）镜像 =="
docker build -f docker/dockerfile --target forecast_api -t drst/forecast_api:latest .

echo "==[3/4] 部署 Online API（FastAPI / Uvicorn，NodePort: 30081）=="
kubectl -n "$NS" apply -f deploy/services/forecast-api.yaml

echo
echo "==[4/4] 提交并运行 Kubeflow Pipeline =="
export KFP_HOST
python3 experiments/kubeflow/submit_pipeline.py

cat <<'EOF'

============ 访问方式 & 查看结果 ============

[在线 API（实时推理服务）]
- 文档/自测(外网)：http://45.149.207.13:30081/docs
- 健康检查：       http://45.149.207.13:30081/
- 示例调用：
  curl -X POST "http://45.149.207.13:30081/predict" \
       -H "Content-Type: application/json" \
       -d '{"inputs":[{"timestamp":"2024-01-01T00:00:00Z","features":{"x1":0.1,"x2":0.2}}]}'

[Kubeflow Pipelines]
- UI：  http://45.149.207.13:30080/pipeline
- 提交脚本会在控制台打印 run url，或到 UI -> Experiments -> 最新运行查看
- DAG 顺序：
  1) offline_training_op（离线特征/初训，产出 baseline 等）
  2) producer_op（向 Kafka 注入数据流）
  3) monitor_op（漂移检测，触发重训）
  4) retrain_op（动态重训 watcher）
  5) infer_op x3（在线推理副本）
  6) forecast_*（时间序列预测：训练/解释/服务）
  7) plot_op（图表与报告、资源汇总）

[MinIO 产物位置]
- s3://onvm-demo2/results/
  · report.md
  · plot_final_timeseries.png / plot_final_relerr.png
  · resources_summary.csv / resources_summary.md
  · 各组件 *_resources.csv（500ms 采样）
- s3://onvm-demo2/models/
  · baseline_model.pt / model.pt / model_*.pt / metrics_*.json

[你需要知道的点]
- Online API 已设置 MINIO_ENDPOINT_URL= http://s3.45.149.207.13.nip.io:30080
  -> API 通过外网网关访问 MinIO；Pipeline 组件仍走集群内 service，二者互不干扰。
- 正常情况下，run_all 结束后：
  · 立刻可访问 API:   http://45.149.207.13:30081/docs
  · Pipeline 运行中： 等待任务完成后，上述 MinIO 路径会出现图/表/报告
============================================
EOF
