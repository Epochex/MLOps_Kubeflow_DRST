#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] line:$LINENO cmd:$BASH_COMMAND"' ERR

: "${NS:=default}"                 # target namespace
: "${RELEASE:=kafka}"              # Helm release name
: "${CHART:=bitnami/kafka}"        # Chart
: "${CHART_VERSION:=}"             # specify chart version (leave empty for latest), e.g., 26.9.0
: "${REPLICAS:=1}"                 # single node is sufficient
: "${HEAP_MB:=1024}"               # Kafka JVM heap size (MiB)

echo "[INFO] namespace=$NS, release=$RELEASE, replicas=$REPLICAS, heap=${HEAP_MB}m"

command -v kubectl >/dev/null || { echo "[FATAL] kubectl not found"; exit 1; }
if ! command -v helm >/dev/null 2>&1; then
  echo "[STEP] install helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"

if ! helm repo list | grep -q '^bitnami'; then
  helm repo add bitnami https://charts.bitnami.com/bitnami
fi
helm repo update >/dev/null

# ============= Generate values.yaml (official image + PLAINTEXT) =============
WORKDIR="$(mktemp -d)"
VALUES="${WORKDIR}/values.yaml"

cat >"$VALUES" <<YAML
# Explicitly specify official image (can be omitted to use chart default)
image:
  registry: docker.io
  repository: bitnami/kafka
  pullPolicy: IfNotPresent

replicaCount: ${REPLICAS}

heapOpts: "-Xmx${HEAP_MB}m -Xms${HEAP_MB}m"

# Disable all authentication, plaintext only (same as your previous values)
listeners:
  client:
    name: CLIENT
    containerPort: 9092
    protocol: PLAINTEXT
    sslClientAuth: none
  controller:
    name: CONTROLLER
    containerPort: 9093
    protocol: PLAINTEXT
    sslClientAuth: none
  interbroker:
    name: INTERNAL
    containerPort: 9094
    protocol: PLAINTEXT
    sslClientAuth: none
  external:
    name: EXTERNAL
    containerPort: 9095
    protocol: PLAINTEXT
    sslClientAuth: none

sasl: {}          # completely disable SASL

service:
  type: ClusterIP  # in-cluster access only

# If you need to fix the number of partitions, set default topic config here (optional)
# kraft:
#   enabled: true  # chart default already enables KRaft (no zookeeper)
YAML

echo "[INFO] rendered values.yaml (head):"
sed 's/^/  /' "$VALUES" | head -n 60

# ============= Install/Upgrade =============
EXTRA_VER=()
[[ -n "$CHART_VERSION" ]] && EXTRA_VER+=(--version "$CHART_VERSION")

echo "[STEP] helm upgrade --install $RELEASE $CHART -n $NS"
helm upgrade --install "$RELEASE" "$CHART" -n "$NS" -f "$VALUES" --wait --timeout 10m "${EXTRA_VER[@]}"

# Wait for StatefulSet Ready
if kubectl -n "$NS" get statefulset "$RELEASE" >/dev/null 2>&1; then
  kubectl -n "$NS" rollout status statefulset "$RELEASE" --timeout=600s
else
  kubectl -n "$NS" get statefulset -l app.kubernetes.io/instance="$RELEASE" || true
fi

# ============= Connection info & smoke test =============
DNS="${RELEASE}.${NS}.svc.cluster.local:9092"
echo
echo " Kafka deployed."
echo "   In-cluster bootstrap:   ${DNS}"
echo "   Example KAFKA_SERVERS:  export KAFKA_SERVERS=${DNS}"
echo

echo "[STEP] smoke test: create topic 'latencyTopic' (idempotent)"
kubectl -n "$NS" exec sts/"$RELEASE" -c kafka -- bash -lc \
  "kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic latencyTopic --partitions 3 --replication-factor 1 || true"

kubectl -n "$NS" exec sts/"$RELEASE" -c kafka -- bash -lc \
  "kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic latencyTopic" || true

echo " delete:  helm -n ${NS} uninstall ${RELEASE}"
