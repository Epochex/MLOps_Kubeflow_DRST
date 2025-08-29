# DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks

This repository supplements our [XX 2025 paper] titled  
**"DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks"**.  

It provides open-source infrastructure to study and reproduce the experiments presented in the paper, including data, scripts, and code for performance inference, forecasting, and drift-aware self-tuning in softwarized NFV environments.

---

## 📘 Overview

### Overview of Infra
This system relies on the Kubernetes architecture and its primary related features as follows:

## Orchestration & Platform Support Chain (Kubeadm/K8s → Calico/StorageClass → Istio/cert-manager → Kubeflow/KFP)
The base layer is a kubeadm-initialized single-node control plane (containerd with SystemdCgroup, swap disabled, `pod-network-cidr` with Calico CNI). The `local-path` StorageClass is set as default for `/mnt/pvc`. Istio provides the ingress gateway and sidecars (NodePort 30080/30443 for UIs); an EnvoyFilter at the gateway injects a user header to bypass login in development, and a DestinationRule disables mTLS toward MinIO. cert-manager is installed with CRDs first and then an Issuer to keep Kubeflow 1.10 install stable; KServe/Knative use server-side apply and CRD readiness waits to avoid race conditions. The pipeline defined in `experiments/kubeflow/pipeline.py` is compiled and submitted to KFP v2; the DAG schedules one pod per stage (offline, monitor, producer, three consumers, plot). An ExitHandler guarantees a report even on failure paths. Components write placeholder metadata to `/tmp/kfp_outputs/`, and metrics are consolidated to object storage for observability and audit, yielding platform-level orchestration with data/model planes cleanly decoupled.

## Data & Model Artifact Plane (MinIO/S3 ↔ PVC/StorageClass ↔ pointer file)
MinIO is the single source of truth for datasets, evaluation outputs, and online results. Pods reach it via `*.svc.cluster.local:9000`, and the Console/API can be exposed through the Istio gateway (Console 9001). The default StorageClass (local-path) backs `/mnt/pvc` inside containers; every artifact is first written locally and then uploaded asynchronously so reads/writes survive network hiccups. Model distribution uses immutable objects plus a mutable pointer: each release uploads timestamped `model_*.pt` and `metrics_*.json`, then overwrites `models/latest.txt` (two lines: model key and metrics key). Consumers read only this pointer for pull-based hot reloads. Traffic to MinIO has an Istio `DestinationRule` that disables mTLS toward the backend to avoid TLS being enforced on a plaintext service. Objects are separated by prefixes `datasets_old/`, `models/`, and `results/` for lifecycle control and audit.

## Streaming & Online Compute Plane (Kafka/KRaft ↔ partitions/consumer groups ↔ Kubernetes parallelism)
Kafka runs in KRaft mode and is exposed as a ClusterIP service; the topic is sharded into three partitions. Multiple Consumer pods join the same consumer group to gain exclusive ownership of partitions, ensuring per-partition ordering and exactly one processor within the group. Producers send a per-partition end-of-stream sentinel; consumers shut down gracefully when the in-process queue drains and an idle timeout elapses, or after all sentinels are seen, then write a trace to storage. Failover and rebalances follow at-least-once semantics; idempotent metrics and final aggregation avoid double-count side effects. Horizontal scale follows “throughput ≈ partitions × schedulable replicas,” with Kubernetes handling placement and cluster DNS providing service discovery, without relying on external load balancers.

## Drift Monitoring → Model Update Control Plane (sliding window → trigger flags → grid retrain → hot reload)
The monitor builds a fixed sliding window from the Kafka stream and, at a configured stride, computes the Jensen–Shannon distance of feature distributions. When thresholds (A/B/C) are crossed, it writes `latest_batch.npy` and `retrain_grid.flag` to object storage and sets a lock to prevent duplicate triggers. The retrainer then runs the corresponding grid search (prefer fine-tune when the structure matches; fall back to from-scratch when needed). After evaluation, it publishes the new model/metrics to MinIO and only updates the pointer, while consumers perform atomic hot reloads via polling, md5 deduplication, and an accuracy-gain threshold. This separates the data plane (Kafka) from the artifact plane (MinIO) and uses flags plus a lock to achieve single-shot triggering with restart safety, with rollback done by pointing `latest.txt` back to a prior model.

![Infra Overview](<docs/structure_infra.png>)

This repository is organized as follows:

- **datasets//**  
  Network-flow datasets. Offline training uses combined.csv to produce the baseline model. Stage 1 uses random_rates.csv for initial adaptation. Stage 2 injects resource_stimulus_global_A-B-C_modified.csv to simulate inference under CPU resource contention.
- **drst_common//**  
  Designed to consolidate four cross-cutting concerns—configuration, artifacts, messaging, and metrics—so monitoring, retraining, and inference are loosely coupled via shared artifacts and flags with low latency. At runtime this layer centralizes rules and artifact access: the config module defines the mapping from drift tiers to retraining grids (and generates those grids), replacing ad-hoc JSON/env handshakes with local writes and flag files. The Kafka wrapper handles reliable produce/consume, partition awareness, and per-partition end-of-stream signaling. The MinIO helper implements “write to local PVC then async upload” to keep training/inference threads off the I/O hot path. The artifacts module standardizes reading/writing models, metrics, selected features, and scalers while maintaining the latest pointer used for online hot reloads and retraining. The metric logger writes all instrumentation to local tables and periodically archives/syncs them. The runtime module exposes readiness signals and pipeline metadata. Utilities include fault-tolerant model deserialization, thresholded accuracy, and distribution-distance calculations; the profiler provides a lightweight timing context.

- **drst_inference/**  
  In offline/, features.py loads the training table (MinIO first, local fallback), performs column cleanup, feature selection, and StandardScaler normalization, and persists selected_feats.json and scaler.pkl. model.py provides a compact MLP regressor with an explicit first layer in_features to simplify input alignment online. train_offline.py trains both the baseline (linear) and adaptive (multi-layer MLP), evaluates MAE/RMSE/acc@0.15, and produces baseline_model.pt plus timestamped model_*.pt as the initial model set for online inference. When the pipeline enters streaming, it runs alongside drst_drift/: in online/, producer.py sends data in phased rates from MinIO to Kafka and emits a per-partition end-of-stream sentinel, while inference_consumer.py launches three parallel consumer workers (to cover multiple partitions). Each consumer auto-stops after a configurable inactivity window (default 60s) to indicate readiness. Once data flows, consumers enter steady-state streaming: they compute the Jensen–Shannon divergence on a sliding window (at the sampling interval) for the retraining controller, run dual-path baseline+adaptive predictions, log error quantiles/throughput/CPU and cumulative accuracy, and periodically check latest.txt to hot-swap to a new model only if it outperforms the baseline by a configured margin; until a swap completes, inference continues on the previous model to preserve stability. Drift detection and retraining are initiated by drst_drift/: when the JS distance triggers a retrain, the system freezes the current window statistics and suppresses new triggers to avoid thrash, then—after retraining succeeds and writes back the new model/metrics—consumers detect the update and switch seamlessly. After the stream naturally drains, the three streaming components (producer/consumer/monitor) exit, and plotting/ takes over: plot_final.py reads each consumer’s *_inference_trace.npz to generate time-series overlays and a relative-error histogram; plot_report.py assembles report.md and uploads all artifacts to results/ in MinIO. Interactions with MinIO are minimized and asynchronous (initial reads during training, local write-through then async upload, and async writes at publish time) to reduce resource contention; within the streaming path, data is kept in memory or on local PVC to cut I/O overhead, increase throughput, and maintain inference continuity during retrain and hot reload.

- **drst_forecasting/**  
  Placeholder and baselines for short-term trend forecasting: baseline_mean.py provides a moving-average baseline to quickly validate whether latency/throughput evolves predictably over time; lstm_placeholder.py sketches a minimal LSTM regressor for future time-series modeling of online error or system state. This package is optional in the main pipeline and can be wired in as an auxiliary predictor in the inference stage or as a forward-error estimator in the monitor.

- **drst_drift/**  
  The monitoring–retraining control loop lives here. The monitor first establishes a reference distribution, then using a fixed window and stride computes histogram-based distribution deltas on the stream. Triggering uses a conservative strategy that compares the latest measurement against the historical baseline and acts on the more severe signal. On trigger, it persists the current window’s feature matrix and any available labels, writes a tier flag per policy, and places a lock to prevent duplicate triggers during the retrain window. The retrainer reads the frozen window and tier, pulls the three-tier grid from configuration, runs a quick warm-up scoring pass to shrink the search space, then performs full training and validation on the remaining candidates. It writes the best model and metrics back to the model store, updates the latest pointer, and removes the lock so the monitor resumes sliding-window evaluation. Interpretability runs as a side path on the same window: it defaults to permutation importance for a fast signal and automatically switches to a kernel-based method when available; results are persisted for audit and reporting but do not affect the main decision loop.

- **experiments/**  
  Reproducible experiments and Kubeflow Pipelines assembly. kubeflow/pipeline.py defines the full KFP DAG — offline → (monitor | producer | infer) → plot — with an 8-minute wall-clock cap per streaming stage and consumer auto-shutdown on idle. submit_pipeline.py compiles drift_demo_v2.yaml and submits it to KFP (KFP host/namespace/experiment are provided via environment variables). yamls/ contains example manifests for observability (e.g., Prometheus and Kafka Exporter). The folder enables a one-command workflow to compile and submit an end-to-end run: offline training, streaming drift detection and adaptive updates, and final charts/report generation.

- **deploy/**  
  Infrastructure and platform automation scripts:  
  Auto_deployment_k8s_kubeflow.sh — Installs Kubernetes 1.32 on a single bare-metal node (containerd runtime, Calico CNI, local-path-provisioner as the default StorageClass), then uses Kustomize to fetch and deploy the full Kubeflow v1.10 stack (cert-manager, Istio with oauth2-proxy and Dex, Knative/KServe, Pipelines, Jupyter, Katib, etc.). The script waits for CRDs/Pods, applies required patches for idempotency and stability, and finally exposes istio-ingressgateway and MinIO via NodePort. It also adjusts policies to keep MinIO reachable (disable Envoy TLS to MinIO; relax peer mTLS to PERMISSIVE), resulting in a fully functional, browser-accessible Kubeflow setup on a single machine.  
  Auto_disable_auth.sh — Strips the authN/authZ chain for a dev/test environment. At the Istio gateway and selected in-namespace entry points, EnvoyFilters inject a fixed user identity (kubeflow-userid and x-goog-authenticated-user-email with the accounts.google.com: prefix). Authorization headers are removed only at the gateway to bypass residual OIDC/JWT checks. In the Kubeflow namespace, an allow-all AuthorizationPolicy is applied; WebApps are set with APP_DISABLE_AUTH=True and configured to read the injected headers; the KFP backend is set with KUBEFLOW_USERID_HEADER=kubeflow-userid and an empty prefix. Sidecar injection is enforced so filters take effect; Lua headers():replace is used for compatibility with your Envoy version. A matching Profile is created for the impersonated user. Net result: any request that reaches the NodePort is treated as the fixed user—suitable only for fast local testing and CI/CD bring-up.  

  - **docker/**  
  Deploys Kafka via the official Bitnami chart in PLAINTEXT/KRaft mode, prints the in-cluster bootstrap address, and runs an idempotent smoke test, aligning with the repo’s default KAFKA_SERVERS. Includes helpers like Auto_clear_pods.yaml for basic cleanup/ops.

---

## 📊 Current limitations
- Current implementation is validated on **Kubernetes + Kubeflow** with **Intel Xeon (Cascade Lake)** servers.  
- Extensions to AMD platforms and other MLOps stacks are left as future work.  
- Lightweight online retraining is currently implemented for regression models; deep RL retraining logic will be released in future updates.  

---

## 📄 Technical Report
An extended version of our paper, with additional results and detailed methodology, is available here:  
👉 **[tech-report.pdf](./docs/tech-report.pdf)**  

---

## 🛠 Usage
The contents of this repository can be freely used for **research and education purposes**.  
If you use our tools or find our work helpful, please cite the following publication:

```bibtex
@article{liu2025drst,
  author    = {Qiong Liu and Jianke Lin and Tianzhu Zhang and Leonardo Linguaglossa.},
  title     = {DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks},
  journal   = {xx},
  year      = {2025}
}
