# DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks

This repository supplements our [XX 2025 paper] titled  
**"DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks"**.  

It provides open-source infrastructure to study and reproduce the experiments presented in the paper, including data, scripts, and code for performance inference, forecasting, and drift-aware self-tuning in softwarized NFV environments.

---

## 📘 Overview

## Overview of Infra
This system relies on the Kubernetes architecture and its primary related features as follows:

#### Streaming & Online Compute Plane    
*Kafka/KRaft partitions, consumer groups, and Kubernetes parallelism*  

**Base Layer**  
The environment is bootstrapped with `kubeadm` on a single-node control plane. The container runtime is containerd with `SystemdCgroup`, swap is disabled to comply with kubelet requirements, and Calico CNI provides the pod network (`pod-network-cidr`). Local persistence is handled by the `local-path` StorageClass, mounted under `/mnt/pvc`, ensuring every pod can access PVC-backed volumes for dataset caching and artifact storage.

**Service Mesh and Ingress**  
Istio provides both ingress gateway and sidecar injection. User-facing UIs are exposed through NodePorts **30080/30443**, enabling external access without a LoadBalancer. At the gateway layer, an EnvoyFilter injects static user headers to bypass authentication in development mode, while a dedicated DestinationRule disables mTLS for MinIO traffic to prevent Envoy from enforcing TLS against a plaintext backend. This setup preserves service-to-service security elsewhere while keeping MinIO compatible.

**Certificates and Stability**  
cert-manager is introduced with CRDs applied first, followed by the Issuer configuration. This sequence prevents dependency races and stabilizes Kubeflow 1.10 deployment. KServe and Knative are installed using server-side apply with CRD readiness checks, ensuring that controllers only proceed once all APIs are available, avoiding startup failures and guaranteeing reproducibility of the deployment.

**Pipeline Execution**  
The end-to-end workflow is encoded in `experiments/kubeflow/pipeline.py` and compiled for KFP v2. The DAG launches one pod per stage:  
- **offline training** for baseline and adaptive models  
- **drift monitor** for sliding-window JS evaluation  
- **Kafka producer** to simulate streaming input  
- **three inference consumers** running in parallel for partition-exclusive processing  
- **plotting/reporting** for final visualization and markdown report generation  

An ExitHandler guarantees that reporting runs even if any upstream stage fails, while each component writes placeholder metadata to `/tmp/kfp_outputs/` for orchestration compliance.

**Observability and Decoupling**  
All metrics are written locally under `/mnt/pvc` and asynchronously uploaded to object storage. This provides durable audit trails and consolidated observability across pods. By decoupling orchestration (pipeline scheduling), the data plane (datasets and streams), and the model plane (training and inference artifacts), the system ensures modularity, robustness under failure conditions, and clean separation of responsibilities.



#### Data & Model Artifact Plane    
*MinIO/S3 ↔ PVC/StorageClass ↔ pointer file*  

**MinIO as Source of Truth**  
MinIO serves as the single source of truth for datasets, evaluation outputs, and online results. Pods access it internally via `*.svc.cluster.local:9000`, while the Console/API can also be exposed through the Istio gateway (Console on 9001).

**Storage and Reliability**  
The default `local-path` StorageClass backs `/mnt/pvc` inside containers. All artifacts are first written locally and then uploaded asynchronously, ensuring read/write resilience against transient network issues.

**Model Distribution**  
Models are distributed using immutable objects plus a mutable pointer: each release uploads timestamped `model_*.pt` and `metrics_*.json`, then overwrites `models/latest.txt` with two lines (model key and metrics key). Consumers read only this pointer, enabling pull-based hot reloads.

**Traffic and Security**  
An Istio `DestinationRule` disables mTLS toward MinIO, preventing Envoy from enforcing TLS on a plaintext backend service.

**Object Organization**  
Data is structured with prefixes `datasets_old/`, `models/`, and `results/`, supporting lifecycle management, separation of concerns, and auditability.


#### Streaming & Online Compute Plane
*Kafka/KRaft ↔ partitions / consumer groups ↔ Kubernetes parallelism*    
**Kafka Runtime**  
Kafka operates in KRaft mode and is deployed as a ClusterIP service within the cluster. The target topic is pre-sharded into three partitions, establishing the parallelism baseline for downstream consumers.

**Consumers and Partition Ownership**  
Consumer pods are launched in parallel under a single consumer group. Kafka’s group coordinator assigns each partition to exactly one consumer, ensuring strict ordering within partitions while preventing duplicate processing. This mechanism guarantees exclusivity of partition ownership across replicas.

**Sentinel Signaling and Graceful Shutdown**  
Producers append an explicit per-partition end-of-stream sentinel. Consumers monitor both the internal processing queue and the count of received sentinels:  
- If all sentinels are observed, the consumer terminates.  
- If no new data arrives beyond the idle timeout, the consumer also exits.  
In either case, the pod finalizes by writing a trace artifact to storage, enabling deterministic downstream analysis.

**Reliability and Processing Semantics**  
Rebalances and failover events adhere to Kafka’s at-least-once delivery semantics. To mitigate side effects, all metrics are designed to be idempotent, and aggregation logic is structured to tolerate replayed events without double counting.

**Horizontal Scalability**  
Effective throughput scales with the formula `throughput ≈ partitions × active replicas`. Partition count defines the maximum parallelism, while Kubernetes schedules replicas onto available nodes and provides DNS-based service discovery. No external load balancer is required, as the Kafka service is resolved directly within the cluster network.


### Drift Monitoring → Model Update Control Plane  
*sliding window → trigger flags → grid retrain → hot reload*  
The monitor maintains a fixed sliding window over the Kafka stream and, at a defined stride, evaluates feature distribution drift via Jensen–Shannon distance. When thresholds (A/B/C) are crossed, it writes `latest_batch.npy` and `retrain_grid.flag` to object storage and sets a lock to prevent duplicate triggers.  

The retrainer consumes these signals to launch grid search: it fine-tunes when the structure is compatible, and falls back to from-scratch training otherwise. After evaluation, it publishes timestamped model and metrics files to MinIO and updates only the mutable pointer, while inference consumers perform atomic hot reloads through periodic polling, md5-based deduplication, and accuracy-gain thresholds.  

This design separates the data plane (Kafka) from the artifact plane (MinIO), relying on flag files and lock semantics to ensure single-shot triggering with restart safety. Rollback is achieved by resetting `latest.txt` to reference a prior model version.


![Infra Overview](<docs/structure_infra.png>)

This repository is organized as follows:

- **datasets/**  
  Network-flow datasets. Offline training uses combined.csv to produce the baseline model. Stage 1 uses random_rates.csv for initial adaptation. Stage 2 injects resource_stimulus_global_A-B-C_modified.csv to simulate inference under CPU resource contention.

  **drst_common** is the runtime backbone that turns the whole system into a low-latency, loosely-coupled graph: rules live as code with env-overrides (drift tiers → on-the-fly A/B/C search spaces), artifacts flow through an **immutable-object + mutable-pointer** pattern (timestamped model/metrics with a 2-line “latest” head for atomic hot reloads and easy rollback), and storage follows **PVC-first write → async S3 upload** so training/inference never stall on the network.  
  Messaging is JSON payloads with partition awareness and per-partition sentinels for deterministic shutdown.  
  Metrics are **local-first CSV/JSONL** append (latency/throughput/cold-start/JS ticks/RTT/CPU%) with batched sync upstream, while tiny readiness flags and minimal pipeline metadata keep orchestration observable without coupling.  

  Utilities close the loop: fault-tolerant model deserialization; thresholded accuracy defined on relative error

  $$
  e = \frac{|\hat y - y|}{\max\bigl(|y|,\varepsilon\bigr)}
  $$

  with

  $$
  \mathrm{accuracy}@\tau = \mathrm{mean}\!\left(e \le \tau\right)
  $$

  Histogram-based Jensen–Shannon divergence for drift

  $$
  \mathrm{JSD}(P\|Q)=\tfrac{1}{2}\,\mathrm{KL}(P\|M)+\tfrac{1}{2}\,\mathrm{KL}(Q\|M),\quad M=\tfrac{1}{2}(P+Q)
  $$

  and a micro profiler that wraps critical sections and emits `runtime_ms` into the same metric stream.  

  The result is a single, cohesive plane—configuration, artifacts, messaging, metrics—where components share only pointers and facts (not state), enabling safe model rollouts, predictable lifecycles, and stable p95 latency under streaming load.  



- **drst_inference/**  
  it drives a dual-path predictor that stays accurate and stable under streaming load.

  After offline training publishes a baseline and an initial adaptive model (plus a 2-line pointer to the current release), the streaming phase spins up three parallel consumers to cover topic partitions.

  Each consumer pulls rows, standardizes them with the shared scaler, aligns inputs to both the baseline and adaptive networks, and emits two predictions per batch.

  It tracks relative-error accuracy with threshold τ=0.2, using

    $$
    e = \frac{|\hat y - y|}{\max\bigl(|y|,\varepsilon\bigr)}
    $$

  logging batch and cumulative accuracy, error quantiles (p50/80/90/95/99), throughput, wall/CPU time, and RTT reconstructed from send/receive timestamps.

  A lightweight guard thread periodically reads the release pointer; a hot-swap occurs only when the new model’s validation gain over baseline exceeds a configured margin (percentage points).

  The swap is lock-safe and atomic: inference continues on the previous handle until the new weights are fully materialized, keeping tail latency flat.

  Termination is deterministic: per-partition sentinels and an idle timeout stop consumers cleanly; on exit, each writes a compact trace (timestamps, truth, baseline, adaptive) for plotting.


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
