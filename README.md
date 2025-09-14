# DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks

This repository supplements our [XX 2025 paper] titled  
**"DRST: a Non-Intrusive Framework for Performance Analysis in Softwarized Networks"**.  

It provides open-source infrastructure to study and reproduce the experiments presented in the paper, including data, scripts, and code for performance inference, forecasting, and drift-aware self-tuning in softwarized NFV environments.

---

## 📘 Overview

## 1.1 Overview of Infra
This system relies on the Kubernetes architecture and its primary related features as follows:

### 1.1.0 Streaming & Online Compute Plane  
*Kafka/KRaft partitions, consumer groups, and Kubernetes parallelism*  

### 1.1.1 Base Layer  
The environment is bootstrapped with `kubeadm` on a single-node control plane. The container runtime is containerd with `SystemdCgroup`, swap is disabled to comply with kubelet requirements, and Calico CNI provides the pod network (`pod-network-cidr`). Local persistence is handled by the `local-path` StorageClass, mounted under `/mnt/pvc`, ensuring every pod can access PVC-backed volumes for dataset caching and artifact storage.

### 1.1.2 Service Mesh and Ingress  
Istio provides both ingress gateway and sidecar injection. User-facing UIs are exposed through NodePorts **30080/30443**, enabling external access without a LoadBalancer. At the gateway layer, an EnvoyFilter injects static user headers to bypass authentication in development mode, while a dedicated DestinationRule disables mTLS for MinIO traffic to prevent Envoy from enforcing TLS against a plaintext backend. This setup preserves service-to-service security elsewhere while keeping MinIO compatible.

### 1.1.3 Certificates and Stability  
cert-manager is introduced with CRDs applied first, followed by the Issuer configuration. This sequence prevents dependency races and stabilizes Kubeflow 1.10 deployment. KServe and Knative are installed using server-side apply with CRD readiness checks, ensuring that controllers only proceed once all APIs are available, avoiding startup failures and guaranteeing reproducibility of the deployment.

### 1.1.4 Pipeline Execution  
The end-to-end workflow is encoded in `experiments/kubeflow/pipeline.py` and compiled for KFP v2. The DAG launches one pod per stage:  
- **offline training** for baseline and adaptive models  
- **drift monitor** for sliding-window JS evaluation  
- **Kafka producer** to simulate streaming input  
- **three inference consumers** running in parallel for partition-exclusive processing  
- **plotting/reporting** for final visualization and markdown report generation  

An ExitHandler guarantees that reporting runs even if any upstream stage fails, while each component writes placeholder metadata to `/tmp/kfp_outputs/` for orchestration compliance.

### 1.1.5 Observability and Decoupling  
All metrics are written locally under `/mnt/pvc` and asynchronously uploaded to object storage. This provides durable audit trails and consolidated observability across pods. By decoupling orchestration (pipeline scheduling), the data plane (datasets and streams), and the model plane (training and inference artifacts), the system ensures modularity, robustness under failure conditions, and clean separation of responsibilities.



## 1.2. Data & Model Artifact Plane    
*MinIO/S3 ↔ PVC/StorageClass ↔ pointer file*  

### 1.2.1 MinIO as Source of Truth  
MinIO serves as the single source of truth for datasets, evaluation outputs, and online results. Pods access it internally via `*.svc.cluster.local:9000`, while the Console/API can also be exposed through the Istio gateway (Console on 9001).

### 1.2.2 Storage and Reliability  
The default `local-path` StorageClass backs `/mnt/pvc` inside containers. All artifacts are first written locally and then uploaded asynchronously, ensuring read/write resilience against transient network issues.

### 1.2.3 Model Distribution  
Models are distributed using immutable objects plus a mutable pointer: each release uploads timestamped `model_*.pt` and `metrics_*.json`, then overwrites `models/latest.txt` with two lines (model key and metrics key). Consumers read only this pointer, enabling pull-based hot reloads.

### 1.2.4 Traffic and Security  
An Istio `DestinationRule` disables mTLS toward MinIO, preventing Envoy from enforcing TLS on a plaintext backend service.

### 1.2.5 Object Organization  
Data is structured with prefixes `datasets_old/`, `models/`, and `results/`, supporting lifecycle management, separation of concerns, and auditability.



## 1.3. Streaming & Online Compute Plane
*Kafka/KRaft ↔ partitions / consumer groups ↔ Kubernetes parallelism*    

### 1.3.1 Kafka Runtime  
Kafka operates in KRaft mode and is deployed as a ClusterIP service within the cluster. The target topic is pre-sharded into three partitions, establishing the parallelism baseline for downstream consumers.

### 1.3.2 Consumers and Partition Ownership  
Consumer pods are launched in parallel under a single consumer group. Kafka’s group coordinator assigns each partition to exactly one consumer, ensuring strict ordering within partitions while preventing duplicate processing. This mechanism guarantees exclusivity of partition ownership across replicas.

### 1.3.3 Sentinel Signaling and Graceful Shutdown  
Producers append an explicit per-partition end-of-stream sentinel. Consumers monitor both the internal processing queue and the count of received sentinels:  
- If all sentinels are observed, the consumer terminates.  
- If no new data arrives beyond the idle timeout, the consumer also exits.  
In either case, the pod finalizes by writing a trace artifact to storage, enabling deterministic downstream analysis.

### 1.3.4 Reliability and Processing Semantics  
Rebalances and failover events adhere to Kafka’s at-least-once delivery semantics. To mitigate side effects, all metrics are designed to be idempotent, and aggregation logic is structured to tolerate replayed events without double counting.

### 1.3.5 Horizontal Scalability  
Effective throughput scales with the formula `throughput ≈ partitions × active replicas`. Partition count defines the maximum parallelism, while Kubernetes schedules replicas onto available nodes and provides DNS-based service discovery. No external load balancer is required, as the Kafka service is resolved directly within the cluster network.



## 1.4. Drift Monitoring → Model Update Control Plane  
*sliding window → trigger flags → grid retrain → hot reload*  

### 1.4.1 Drift Detection and Triggers  
The monitor maintains a fixed sliding window over the Kafka stream and, at a defined stride, evaluates feature distribution drift via Jensen–Shannon distance. When thresholds (A/B/C) are crossed, it writes `latest_batch.npy` and `retrain_grid.flag` to object storage and sets a lock to prevent duplicate triggers.  

### 1.4.2 Retraining and Publishing  
The retrainer consumes these signals to launch grid search: it fine-tunes when the structure is compatible, and falls back to from-scratch training otherwise. After evaluation, it publishes timestamped model and metrics files to MinIO and updates only the mutable pointer, while inference consumers perform atomic hot reloads through periodic polling, md5-based deduplication, and accuracy-gain thresholds.  

### 1.4.3 Separation of Planes & Rollback  
This design separates the data plane (Kafka) from the artifact plane (MinIO), relying on flag files and lock semantics to ensure single-shot triggering with restart safety. Rollback is achieved by resetting `latest.txt` to reference a prior model version.


![Infra Overview](<docs/structure_infra.png>)


---

## 2. Software Overview
**The repository in software vision is organized as follows.** 

### 2.1 `datasets/`

> This directory applies to the linear topology only; DAG-1 and DAG-2 are analogous. Which stores **CSV files directly consumable by the system** (either preprocessed/merged results, or 12-row excerpts from each CSV for integration testing).  
> Raw bundles are stored in MinIO under `raw/`. Processed artifacts live in `datasets/` (online inference branch, **perf**) and `datasets_pcm/` (forecast branch, **pcm**).

---

#### Scenarios and Usage (scenarios → explanation)

| scenarios | explanation |
|---|---|
| **stage0 – baseline (offline)** → `datasets/combined.csv` | **Offline training set** for regular traffic (linear topology). The `offline` component trains the baseline on this file and produces `feature_cols.json / model.pt` under `models/`. |
| **stage1 – random rates** → `datasets/stage1_random_rates.csv` | **Traffic-only stimulation** (random rates). Derived from VNF + TX/RX + Latency raw files in `raw/random_rates/exp-*/`, merged by **perf** preprocessing. Used by the Producer as the input for the **first online adaptation**. |
| **stage2 – resource stimulus** → `datasets/stage2_resource_stimulus_global_A-B-C_modified.csv` | **CPU resource contention only** (no additional traffic changes). Derived from `raw/resource_stimulus/`, preprocessed by **perf**, then filtered to the **A-B-C_modified** subset to emulate deployments under resource contention. |
| **stage3 – intervention** → `datasets/stage3_intervention_global.csv` | **Most complex intervention**: **traffic variation + resource contention** simultaneously. Derived from `raw/intervention/`, merged via **perf** preprocessing. Used to evaluate robustness under extreme disturbances. |
| **forecasting (PCM branch)** → `datasets_pcm/*.csv` | Used **only** by the **time-series forecasting branch** (Forecast-GridSearch / XAI). Raw bundles in `raw/pcm/const-?gbps/` are turned into sequences via **pcm** preprocessing/extraction. Runs **in parallel** with the online inference branch, with **no mutual dependency**. |

---

#### Raw Bundles Sources

| scenarios | explanation |
|---|---|
| `random_rates` | Source of Stage 1: multiple `exp-*` experiment folders containing `perf stat` counters for 5 VNFs (`firewall.csv`, `nf_router.csv`, `ndpi_stats.csv`, `payload_scan.csv`, `bridge.csv`) plus `tx_stats.csv`, `rx_stats.csv`, and `latency.csv`. Merged by **perf** preprocessing → `datasets/stage1_random_rates.csv`. |
| `resource_stimulus` | Source of Stage 2: various CPU stress patterns. Preprocessed by **perf** → merged, then filtered to **A-B-C_modified** → `datasets/stage2_resource_stimulus_global_A-B-C_modified.csv`. |
| `intervention` | Source of Stage 3: complex interventions that change both traffic and resources. Preprocessed by **perf** → `datasets/stage3_intervention_global.csv`. |
| `pcm_const_*` (or `pcm/const-?gbps/`) | Forecast branch: multiple constant-rate experiments (with richer hardware counters such as CPU/LLC/memory/PCIe). Preprocessed/extracted by **pcm** → `datasets_pcm/*.csv` for time-series forecasting and explanation. |
| `load_stimulus` / `fixed_rate.zip` / `incremental.zip` | Additional **traffic-stimulation** or historical experiment materials. They can be normalized by the **perf** toolchain into CSVs with unified headers; **not included by default** in the three main stages above—enable/switch in Producer as needed. |

---

#### Meaning of Raw CSVs (Perf example)

- **VNF counter files**: `firewall.csv`, `nf_router.csv`, `ndpi_stats.csv`, `payload_scan.csv`, `bridge.csv`  
  - Sourced from `perf stat` (comma-separated), each row advances in time; during preprocessing, events are matched by full-line event name and the **largest-magnitude** numeric value on that line is taken as the event count (to avoid picking the `scale≈1.x` field).  
  - Event set (12 items):  
    `instructions, branches, branch-misses, branch-load-misses, cache-misses, cache-references, cycles, L1-dcache-load-misses, L1-dcache-loads, LLC-load-misses, LLC-stores, LLC-loads`  
  - Columns after preprocessing: `\<vnf>_\<event>` (e.g., `firewall_instructions`).

- **TX/RX traffic files**: `tx_stats.csv`, `rx_stats.csv`  
  - If a header exists, prefer columns `Mbit` or `PacketRate`; otherwise fall back to fixed positional parsing.  
  - Preprocessing aligns them to KPIs: `input_rate` and `output_rate` (units follow the source columns: typically **Mb/s** or **pps**).

- **End-to-end latency**: `latency.csv` (or `latency_old.csv`)  
  - One latency per row; units are auto-detected: if large numeric scales are detected, convert **μs → ms**.  
  - Exposed as KPI: `latency` (milliseconds) after preprocessing.

- **Alignment strategy**: use the length of `firewall_instructions` as the anchor (fallback to TX/RX/Latency if missing) and truncate/pad all columns to the **same sequence length**; fill missing values with `NaN`.

- **Output column schema** (uniform **63 columns**):  
  `["input_rate", "output_rate", "latency"] + [f"{vnf}_{event}" for vnf in (firewall, nf_router, ndpi_stats, payload_scan, bridge) for event in the 12 events]`

> The above processing is implemented by `drst_preprocess/perf/preprocess_perf.py` (per-experiment parsing/alignment) and `drst_preprocess/perf/extract_perf.py` (merging within the same scenario).




### 2.2 `drst_common/`
The runtime backbone that turns the whole system into a low-latency, loosely-coupled graph: rules live as code with env-overrides (drift tiers → on-the-fly A/B/C search spaces), artifacts flow through an **immutable-object + mutable-pointer** pattern (timestamped model/metrics with a 2-line “latest” head for atomic hot reloads and easy rollback), and storage follows **PVC-first write → async S3 upload** so training/inference never stall on the network.  
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

### 2.3 `drst_inference/`
#### 2.3.1 Execution Path & Parallel Consumption
The offline phase produces a baseline model and an initial adaptive model, and publishes a small “release pointer” that designates the active version. The streaming phase then launches parallel consumers matched to the topic’s partitions. Each consumer pulls records, standardizes them with a shared scaler, aligns inputs to both the baseline and adaptive networks, and emits **two predictions per batch**. Input dimensions are strictly reconciled with each network’s first layer (padding or truncation as needed) so the two paths remain comparable on the same data.

#### 2.3.2 Online Metrics & Observability
The system continuously tracks batch-level and cumulative accuracy at a fixed relative-error threshold of 0.2, and logs error quantiles (p50/80/90/95/99), throughput, wall-clock latency, and CPU time. End-to-end RTT is reconstructed from send/receive timestamps to expose network jitter. A lightweight timer wraps critical sections and emits `runtime_ms` into the same metric stream. All metrics are appended locally first and only then batched upstream, ensuring observability never back-pressures inference under load.

#### 2.3.3 Hot Swap & Latency Control
A background guard thread periodically reads the release pointer. A model swap is triggered only when the **validation gain over the baseline** exceeds a configured margin (in percentage points). The swap is an atomic pointer replacement: loading and verification occur under a narrow lock, while inference threads continue to serve on the previous handle until the new weights are fully materialized, keeping tail latency flat. If fetching fails or the gain is insufficient, the update is skipped and the current model remains in service.

#### 2.3.4 Deterministic Shutdown & Closed-Loop Data
Termination is deterministic: the consumers exit cleanly once **all per-partition sentinels** have arrived or an extended idle timeout is reached. On exit, each consumer writes a compact time-series trace (timestamps, ground truth, baseline prediction, adaptive prediction) for downstream aggregation and plotting. This stitches the offline bridge and the online phases onto a single timeline, enabling full replay of the experiment and phase-level diagnosis.

### 2.4 `drst_forecasting/`
Placeholder and baselines for short-term trend forecasting: `baseline_mean.py` provides a moving-average baseline to quickly validate whether latency/throughput evolves predictably over time; `lstm_placeholder.py` sketches a minimal LSTM regressor for future time-series modeling of online error or system state. This package is optional in the main pipeline and can be wired in as an auxiliary predictor in the inference stage or as a forward-error estimator in the monitor.

### 2.5 `drst_drift/`
#### 2.5.1 Monitor — continuous sensing & safe triggering
A lightweight listener threads records into an in-memory queue; the main loop standardizes features and advances a fixed-size sliding window by a regular stride. It builds an initial baseline window and freezes per-feature histogram ranges from that snapshot so later comparisons aren’t skewed by moving bins. At each step it computes a per-feature histogram JSD to the baseline and averages them into one drift signal. Triggering is data-driven: thresholds for A/B/C are bootstrap-calibrated from the baseline’s own variability (quantile cutoffs), not hand-tuned or smoothed by moving means. On trigger the monitor snapshots the entire current window (features + any labels), writes a tier flag, and sets a lock to suppress duplicate triggers while retraining runs. When the retrainer writes a completion flag, the lock is released and—by policy—the baseline is refreshed to that last window and thresholds are re-calibrated, letting “normal” follow the new regime. Deterministic shutdown uses partition sentinels plus an idle timeout, and metrics are local-first with batched, async uplink.

#### 2.5.2 Retrainer — tiered search & atomic rollout
The retrainer consumes the frozen window and tier flag, reuses the offline scaler and selected features, and makes a train/validation split. It loads the current latest model and, when architecture/activation are compatible, fine-tunes; otherwise it trains from scratch. A centralized config provides the A/B/C grids. The retrainer does a quick warm-up scoring to rank candidates, then fully trains only the shortlist with early stopping, selecting the best by validation error. It writes model bytes and metrics, atomically updates the “latest” pointer for online hot-reload, and emits a retrain-done flag that unlocks the monitor. Detection stays histogram-JSD with frozen bins and calibrated thresholds for stability and interpretability; adaptation focuses on the latest window, keeping responses sharp without chasing noise.

### 2.6 `experiments/`
Reproducible experiments and Kubeflow Pipelines assembly. `kubeflow/pipeline.py` defines the full KFP DAG — offline → (monitor | producer | infer) → plot — with an 8-minute wall-clock cap per streaming stage and consumer auto-shutdown on idle. `submit_pipeline.py` compiles `drift_demo_v2.yaml` and submits it to KFP (KFP host/namespace/experiment are provided via environment variables). `yamls/` contains example manifests for observability (e.g., Prometheus and Kafka Exporter). The folder enables a one-command workflow to compile and submit an end-to-end run: offline training, streaming drift detection and adaptive updates, and final charts/report generation.
![Pipeline Overview](<docs/pipeline_runtime.png>)


### 2.7 `deploy/`
#### 2.7.1 `Auto_deployment_k8s_kubeflow.sh`

Provision a single-node **Kubernetes v1.32** cluster (containerd runtime, **Calico** CNI, `local-path` as the default `StorageClass`), then fetch and deploy **Kubeflow v1.10** via Kustomize (**cert-manager** with issuer, **Istio** + oauth2-proxy + **Dex**, **Knative/KServe**, **Pipelines**, **Jupyter**, **Katib**, **Training Operator**, etc.).  
The script waits for CRDs/Pods, applies idempotent patches, and exposes **istio-ingressgateway** as **NodePort 30080/30443**.

**Object storage normalization.** MinIO is pinned and exposed; Envoy `DestinationRule`s **disable TLS toward MinIO** to avoid TLS to a plaintext backend, and the script **creates the `mlpipeline` bucket** so KFP drivers don’t fail with “bucket does not exist”.

**KFP auth-bypass + UI status-sync (test-only).**
- Inject a fixed user at the **gateway** (`kubeflow-userid`, `x-auth-request-email`, `x-goog-authenticated-user-email`) and strip `Authorization`.
- Add an **inbound fallback header** filter on the `ml-pipeline` sidecar.
- Disable app-level auth on all Kubeflow WebApps and **unify the user header**; set the KFP backend to the same header and make the **persistenceagent watch all namespaces**.
- **Disable Istio sidecar** on MySQL to prevent DB connectivity issues.

**RBAC_MODE switch**
- **A** – grant the fixed user **cluster-admin** (quick & dirty for tests).
- **B (default)** – minimal RBAC: a `ClusterRole` that allows `pipelines.kubeflow.org/workflows` with `verbs: ["*"]`, so the backend’s SAR (`report` check) passes.  
This resolves the **“runs stay Pending in UI”** problem by allowing the persistence agent to write real run states back to the DB.

> **Usage**
> ```bash
> USER_EMAIL=user@example.com RBAC_MODE=B ./deploy/Auto_deployment_k8s_kubeflow.sh
> ```
> ⚠️ Auth-bypass is for **development/testing only**. For production, remove the EnvoyFilters, re-enable WebApp auth, and integrate with your real IdP/RBAC.

![Infra Overview](<docs/KubeflowWebUI.png>)

#### 2.7.2 `Auto_disable_auth.sh`
Strips the authN/authZ chain for a dev/test environment. At the Istio gateway and selected in-namespace entry points, EnvoyFilters inject a fixed user identity (`kubeflow-userid` and `x-goog-authenticated-user-email` with the `accounts.google.com:` prefix). Authorization headers are removed only at the gateway to bypass residual OIDC/JWT checks. In the Kubeflow namespace, an allow-all AuthorizationPolicy is applied; WebApps are set with `APP_DISABLE_AUTH=True` and configured to read the injected headers; the KFP backend is set with `KUBEFLOW_USERID_HEADER=kubeflow-userid` and an empty prefix. Sidecar injection is enforced so filters take effect; Lua `headers():replace` is used for compatibility with your Envoy version. A matching Profile is created for the impersonated user. Net result: any request that reaches the NodePort is treated as the fixed user—suitable only for fast local testing and CI/CD bring-up.

#### 2.7.3 `Auto_deploy_kafka.sh`
Installs Kafka bootstrap via Bitnami Helm. Ensures Helm and target namespace, adds and updates repo, renders a minimal `values.yaml`: single-broker (tunable), PLAINTEXT listeners (no SASL/TLS), KRaft (chart default), ClusterIP Service, configurable JVM heap. Installs and upgrades the release, waits for rollout, prints the in-cluster bootstrap address, and smoke-creates `latencyTopic` (3 partitions).



#### 2.8 `docker/`
Deploys Kafka via the official Bitnami chart in PLAINTEXT/KRaft mode, prints the in-cluster bootstrap address, and runs an idempotent smoke test, aligning with the repo’s default `KAFKA_SERVERS`. Includes helpers like `Auto_clear_pods.yaml` for basic cleanup/ops.

![Pipeline Overview](<docs/pipeline_runtime3.png>)

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
