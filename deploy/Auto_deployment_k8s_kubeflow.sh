#!/usr/bin/env bash

MEM_KB=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
REQ_KB=$((24 * 1024 * 1024))    
MEM_GIB=$(( (MEM_KB + 1024*1024 - 1) / (1024*1024) ))  
echo "make sure that you are in the folder you wana deploy these system and platform"
echo "Detected -> RAM: ~${MEM_GIB} GiB"


if [[ "${AUTO_CONFIRM:-}" != "y" ]]; then
  read -r -p "Proceed ONLY if this host is single-NIC AND RAM >= 24GiB. Are you sure and continue? [y/N]: " REPLY
  REPLY=${REPLY:-N}  # Enter as NO
  if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "byebye."; exit 1
  fi
fi

set -Eeuo pipefail
trap 'echo "[ERR] line:$LINENO cmd:$BASH_COMMAND"' ERR
export DEBIAN_FRONTEND=noninteractive

echo -e "\033[1;31m==> Deploying Kubernetes Bare Metal Standalone and related components ...\033[0m"


# Basic dependencies
apt update
# Prefer containerd.io; if the repo is unavailable, fall back to distro containerd
if ! apt install -y containerd.io; then
  apt install -y containerd
fi


# Disable swap (permanently)
swapoff -a
sed -i.bak '/\sswap\s/s/^/#/' /etc/fstab

# Kernel modules & sysctl
cat >/etc/modules-load.d/k8s.conf <<'EOF'
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter

cat >/etc/sysctl.d/99-k8s.conf <<'EOF'
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
sysctl --system

# Install containerd (Docker repo, with full CRI)
apt update
apt install -y containerd.io

# Generate and rewrite config: systemd cgroups + pause 3.10
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sed -i 's#sandbox_image = "registry.k8s.io/pause:3\..*"#sandbox_image = "registry.k8s.io/pause:3.10"#' /etc/containerd/config.toml
systemctl enable --now containerd

# crictl points to containerd
cat >/etc/crictl.yaml <<'EOF'
runtime-endpoint: unix:///var/run/containerd/containerd.sock
image-endpoint: unix:///var/run/containerd/containerd.sock
timeout: 10
debug: false
EOF

# Install kubelet/kubeadm/kubectl (K8s 1.32 stable repo)
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.32/deb/Release.key \
  | gpg --dearmor -o /etc/apt/keyrings/kubernetes-1-32.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-1-32.gpg] https://pkgs.k8s.io/core:/stable:/v1.32/deb/ /" \
  >/etc/apt/sources.list.d/kubernetes-1-32.list

apt update
apt install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl
systemctl enable kubelet

# Pre-pull images (will auto-select current repo 1.32.x patch version)
kubeadm config images pull

# Initialize single-node control plane (aligned with Calico default subnet)
kubeadm init --pod-network-cidr=192.168.0.0/16

# Configure kubectl
mkdir -p $HOME/.kube
cp /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

# Allow scheduling on control-plane (single node)
kubectl taint nodes --all node-role.kubernetes.io/control-plane- || true

# Install Calico v3.30.x (supports K8s 1.32/1.33)
curl -L https://raw.githubusercontent.com/projectcalico/calico/v3.30.2/manifests/calico.yaml -o calico.yaml
kubectl apply -f calico.yaml

# Wait for nodes Ready
kubectl wait --for=condition=Ready node --all --timeout=300s

# Install default StorageClass (local-path-provisioner)
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
kubectl -n local-path-storage rollout status deploy/local-path-provisioner --timeout=180s
kubectl patch storageclass local-path -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

echo -e "\033[1;31m==> Kubernetes deployment completed, deploying Kubeflow system...\033[0m"


set -euxo pipefail

# Install kustomize (v5.5.0+)
KVER=v5.5.0
curl -L -o kustomize.tar.gz https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F${KVER}/kustomize_${KVER}_linux_amd64.tar.gz
tar -xzf kustomize.tar.gz
install -m 0755 kustomize /usr/local/bin/kustomize
kustomize version

# Pull Kubeflow manifests v1.10-branch (do not use master)
rm -rf manifests 2>/dev/null || true
git clone --depth=1 --branch v1.10-branch https://github.com/kubeflow/manifests.git manifests
cd manifests

# 2.1 cert-manager
kustomize build common/cert-manager/base | kubectl apply -f -

# Wait for CRDs Established (avoid CRD/CR race conditions)
for crd in \
  certificaterequests.cert-manager.io \
  certificates.cert-manager.io \
  challenges.acme.cert-manager.io \
  clusterissuers.cert-manager.io \
  issuers.cert-manager.io \
  orders.acme.cert-manager.io; do
  kubectl wait --for=condition=Established crd/$crd --timeout=180s || true
done

# Wait for cert-manager Deployments ready
kubectl -n cert-manager rollout status deploy/cert-manager --timeout=300s
kubectl -n cert-manager rollout status deploy/cert-manager-webhook --timeout=300s
kubectl -n cert-manager rollout status deploy/cert-manager-cainjector --timeout=300s

# Wait for webhook backend Endpoints (Service connectivity)
until kubectl -n cert-manager get endpoints cert-manager-webhook -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null | grep -q .; do
  echo "[cert-manager] waiting for webhook endpoints ..."; sleep 3
done

# Then create issuer, retry until success (avoid set -e abort)
until kustomize build common/cert-manager/kubeflow-issuer/base | kubectl apply -f -; do
  echo "[issuer] webhook not ready yet, retrying in 5s ..."; sleep 5
done


# 2.2 Istio (using oauth2-proxy option)
kustomize build common/istio/istio-crds/base | kubectl apply -f -
kustomize build common/istio/istio-namespace/base | kubectl apply -f -
kustomize build common/istio/istio-install/overlays/oauth2-proxy | kubectl apply -f -
kubectl -n istio-system wait --for=condition=Ready pods --all --timeout=300s

# 2.3 oauth2-proxy (select m2m-dex-only, generic)
kustomize build common/oauth2-proxy/overlays/m2m-dex-only | kubectl apply -f -
kubectl -n oauth2-proxy wait --for=condition=Ready pod -l app.kubernetes.io/name=oauth2-proxy --timeout=180s

# 2.4 Dex (default user user@example.com / 12341234, change password/email for production)
kustomize build common/dex/overlays/oauth2-proxy | kubectl apply -f -
kubectl -n auth wait --for=condition=Ready pods --all --timeout=180s

# 2.5 Knative (KServe dependency) — install first, wait CRDs, rerun after ready
kustomize build common/knative/knative-serving/overlays/gateways | kubectl apply -f -

# Wait for CRDs Established (avoid CRD/CR race conditions)
for crd in $(kubectl get crd -o name | grep -E 'knative\.dev|caching\.internal\.knative\.dev|networking\.internal\.knative\.dev'); do
  kubectl wait --for=condition=Established $crd --timeout=180s || true
done

# Apply again (first run often shows “no matches for kind”, second run succeeds)
kustomize build common/knative/knative-serving/overlays/gateways | kubectl apply -f -

# Install cluster-local-gateway
kustomize build common/istio/cluster-local-gateway/base | kubectl apply -f -

# Knative Serving components all ready
kubectl -n knative-serving wait --for=condition=Ready pod -l 'app in (controller,autoscaler,webhook,net-istio-controller,net-istio-webhook,activator)' --timeout=300s || true


# 2.6 Base namespaces/roles/policies/istio resources
kustomize build common/kubeflow-namespace/base | kubectl apply -f -
kustomize build common/networkpolicies/base | kubectl apply -f -
kustomize build common/kubeflow-roles/base | kubectl apply -f -
kustomize build common/istio/kubeflow-istio-resources/base | kubectl apply -f -

# 2.6.x Metacontroller (KFP profile-controller dependency) — fixed upstream version, avoid remote direct link
MC_REF=v4.11.0

# Namespace & disable Istio injection (avoid sidecar interference)
kubectl get ns metacontroller >/dev/null 2>&1 || kubectl create ns metacontroller
# Use patch to ensure label applied, avoid “not labeled”
kubectl patch ns metacontroller --type=merge -p '{"metadata":{"labels":{"istio-injection":"disabled"}}}'

# Clone fixed version via git, avoid remote kustomize direct link
rm -rf /tmp/metacontroller-${MC_REF} 2>/dev/null || true
git clone --depth=1 --branch ${MC_REF} https://github.com/metacontroller/metacontroller /tmp/metacontroller-${MC_REF}

# Apply production manifests
kustomize build /tmp/metacontroller-${MC_REF}/manifests/production \
  | kubectl -n metacontroller apply -f -

# Wait for CRDs Ready (avoid “no matches for kind”)
kubectl wait --for=condition=Established crd/decoratorcontrollers.metacontroller.k8s.io --timeout=180s || true
kubectl wait --for=condition=Established crd/compositecontrollers.metacontroller.k8s.io --timeout=180s || true

# Wait workload depending on resource type (v4.11.0 uses StatefulSet)
if kubectl -n metacontroller get deploy/metacontroller >/dev/null 2>&1; then
  kubectl -n metacontroller rollout status deploy/metacontroller --timeout=300s
elif kubectl -n metacontroller get statefulset/metacontroller >/dev/null 2>&1; then
  kubectl -n metacontroller rollout status statefulset/metacontroller --timeout=300s
else
  echo "[metacontroller] workload not found"; kubectl -n metacontroller get all; exit 1
fi

# Extra wait for Pod Ready (sometimes rollout done but pods still pulling images)
kubectl -n metacontroller wait --for=condition=Ready pod -l app=metacontroller --timeout=300s || true



# 2.7 Kubeflow Pipelines (multi-user, runtime 2.5.0)
kustomize build applications/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user | kubectl apply -f -

# 2.7.x KServe (use Server-Side Apply to avoid oversized last-applied annotations)
while ! kustomize build applications/kserve/kserve \
  | kubectl apply --server-side --force-conflicts -f - ; do
  echo "[kserve] waiting CRDs to register..."; sleep 5
done

# Explicitly wait for KServe CRDs Established (avoid CRD/CR race)
for crd in \
  inferenceservices.serving.kserve.io \
  trainedmodels.serving.kserve.io \
  servingruntimes.serving.kserve.io \
  clusterservingruntimes.serving.kserve.io \
  clusterstoragecontainers.serving.kserve.io; do
  kubectl wait --for=condition=Established crd/$crd --timeout=180s || true
done

# Re-apply again (still with Server-Side) to ensure ClusterServingRuntime/ClusterStorageContainer created
kustomize build applications/kserve/kserve \
  | kubectl apply --server-side --force-conflicts -f -



# 2.8 Other components (install as needed)
kustomize build applications/kserve/kserve | kubectl apply --server-side --force-conflicts -f -
kustomize build applications/kserve/models-web-app/overlays/kubeflow | kubectl apply -f -
kustomize build applications/katib/upstream/installs/katib-with-kubeflow | kubectl apply -f -
kustomize build applications/centraldashboard/overlays/oauth2-proxy | kubectl apply -f -
kustomize build applications/admission-webhook/upstream/overlays/cert-manager | kubectl apply -f -
kustomize build applications/jupyter/notebook-controller/upstream/overlays/kubeflow | kubectl apply -f -
kustomize build applications/jupyter/jupyter-web-app/upstream/overlays/istio | kubectl apply -f -
kustomize build applications/pvcviewer-controller/upstream/base | kubectl apply -f -
kustomize build applications/profiles/upstream/overlays/kubeflow | kubectl apply -f -
kustomize build applications/volumes-web-app/upstream/overlays/istio | kubectl apply -f -
kustomize build applications/tensorboard/tensorboards-web-app/upstream/overlays/istio | kubectl apply -f -
kustomize build applications/tensorboard/tensorboard-controller/upstream/overlays/kubeflow | kubectl apply -f -
kustomize build applications/training-operator/upstream/overlays/kubeflow | kubectl apply --server-side --force-conflicts -f -



# 2.9 Create default user namespace
kustomize build common/user-namespace/base | kubectl apply -f -

# 10) (HTTP access) disable secure cookies + disable KFP UI metadata probing
kubectl -n kubeflow set env deploy/centraldashboard APP_SECURE_COOKIES=false || true
kubectl -n kubeflow set env deploy/jupyter-web-app APP_SECURE_COOKIES=false || true
kubectl -n kubeflow set env deploy/tensorboards-web-app APP_SECURE_COOKIES=false || true
kubectl -n kubeflow set env deploy/volumes-web-app APP_SECURE_COOKIES=false || true
kubectl -n kubeflow set env deploy/ml-pipeline-ui DISABLE_GKE_METADATA=true || true

# 2.10 Wait for key Pods Ready (longer timeout)
for ns in istio-system cert-manager auth oauth2-proxy knative-serving kubeflow kubeflow-user-example-com; do
  kubectl -n "$ns" get pod || true
done

echo -e "\033[1;31m==> Deployment is complete. Please wait for the relevant pods to restart automatically before normal use.\033[0m"
echo -e "\033[1;31m==> Estimated time: 5 minutes.\033[0m"

kubectl -n istio-system patch svc istio-ingressgateway -p '{
  "spec": {
    "type": "NodePort",
    "ports": [
      {"name":"http2","port":80,"targetPort":8080,"nodePort":30080},
      {"name":"https","port":443,"targetPort":8443,"nodePort":30443}
    ]
  }
}'
kubectl -n istio-system get svc istio-ingressgateway -o wide

echo -e "\033[1;31m==> Kubeflow has been exposed to port 30080. You can access the Kubeflow UI by entering ip:30080 in your browser.\033[0m"




###############################################################################
# MinIO —— Enables 9000(S3)/9001(Console) under Istio and provides a browser entry point
###############################################################################

echo "[minio] configuring deployment (image/console/root creds/istio off)..."
# Upgrade/Standardize MinIO (still using the default Kubeflow secret: mlpipeline-minio-artifact)
kubectl -n kubeflow patch deploy/minio --type='json' -p='[
  {"op":"add","path":"/spec/template/metadata/annotations/sidecar.istio.io~1inject","value":"false"},
  {"op":"replace","path":"/spec/template/spec/containers/0/image","value":"quay.io/minio/minio:RELEASE.2022-10-29T06-21-33Z"},
  {"op":"replace","path":"/spec/template/spec/containers/0/args","value":["server","/data","--console-address",":9001"]},
  {"op":"add","path":"/spec/template/spec/containers/0/env/-","value":{"name":"MINIO_ROOT_USER","valueFrom":{"secretKeyRef":{"name":"mlpipeline-minio-artifact","key":"accesskey"}}}},
  {"op":"add","path":"/spec/template/spec/containers/0/env/-","value":{"name":"MINIO_ROOT_PASSWORD","valueFrom":{"secretKeyRef":{"name":"mlpipeline-minio-artifact","key":"secretkey"}}}}
]' || true

echo "[minio] normalizing Service ports (names/NodePort keep)..."
if kubectl -n kubeflow get svc minio-service >/dev/null 2>&1; then
  kubectl -n kubeflow patch svc minio-service --type='json' -p='[
    {"op":"replace","path":"/spec/type","value":"NodePort"},
    {"op":"replace","path":"/spec/ports/0/name","value":"http-api"}
  ]' || true
  if kubectl -n kubeflow get svc minio-service -o jsonpath='{.spec.ports[1].port}' >/dev/null 2>&1; then
    kubectl -n kubeflow patch svc minio-service --type='json' -p='[
      {"op":"replace","path":"/spec/ports/1/name","value":"http-console"},
      {"op":"replace","path":"/spec/ports/1/port","value":9001},
      {"op":"replace","path":"/spec/ports/1/targetPort","value":9001}
    ]' || true
  else
    kubectl -n kubeflow patch svc minio-service --type='json' -p='[
      {"op":"add","path":"/spec/ports/-","value":{"name":"http-console","port":9001,"targetPort":9001,"protocol":"TCP","nodePort":30901}}
    ]' || true
  fi
fi

echo "[minio] disable TLS-to-MinIO (ingress & in-namespace) and relax mTLS on MinIO pods..."
# Disable TLS towards MinIO (prevent Envoy from handshaking TLS to plaintext backend) and set MinIO to PERMISSIVE

cat <<'YAML' | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: minio-backend-plain
  namespace: istio-system
spec:
  host: minio-service.kubeflow.svc.cluster.local
  trafficPolicy:
    tls:
      mode: DISABLE
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: minio-backend-plain
  namespace: kubeflow
spec:
  host: minio-service.kubeflow.svc.cluster.local
  trafficPolicy:
    tls:
      mode: DISABLE
--



###############################################################################
# KFP auth-bypass + UI status sync (switchable RBAC_MODE)
###############################################################################
set -Eeuo pipefail

USER_EMAIL=${USER_EMAIL:-user@example.com}           # fix the login user, same as gateway
RBAC_MODE=${RBAC_MODE:-A}                            # A=cluster-admin  B=minimal
NS_KF=${NS_KF:-kubeflow}
NS_ISTIO=${NS_ISTIO:-istio-system}
NS_USER=${NS_USER:-kubeflow-user-example-com}

echo "[kfp] allow-all AuthorizationPolicy in ${NS_KF} ..."
cat <<YAML | kubectl apply -f -
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: allow-all-kubeflow
  namespace: ${NS_KF}
spec:
  rules:
  - {}
YAML

echo "[kfp] EnvoyFilter@gateway: force user headers + strip Authorization ..."
cat <<YAML | kubectl apply -f -
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: force-kubeflow-user
  namespace: ${NS_ISTIO}
spec:
  workloadSelector:
    labels:
      app: istio-ingressgateway
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: GATEWAY
      listener:
        filterChain:
          filter:
            name: envoy.filters.network.http_connection_manager
            subFilter:
              name: envoy.filters.http.router
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.lua
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
          inline_code: |
            function envoy_on_request(handle)
              local uid = "${USER_EMAIL}"
              local h = handle:headers()
              h:replace("kubeflow-userid", uid)
              h:replace("x-auth-request-email", uid)
              h:replace("x-goog-authenticated-user-email", "accounts.google.com:" .. uid)
              h:remove("authorization")
            end
YAML

echo "[kfp] EnvoyFilter@ml-pipeline inbound: fallback user headers ..."
cat <<YAML | kubectl apply -f -
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: add-userid-to-ml-pipeline-inbound
  namespace: ${NS_KF}
spec:
  workloadSelector:
    labels:
      app: ml-pipeline
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
      listener:
        filterChain:
          filter:
            name: envoy.filters.network.http_connection_manager
            subFilter:
              name: envoy.filters.http.router
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.lua
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
          inline_code: |
            function envoy_on_request(handle)
              local uid = "${USER_EMAIL}"
              local h = handle:headers()
              if not h:get("kubeflow-userid") then h:replace("kubeflow-userid", uid) end
              if not h:get("x-auth-request-email") then h:replace("x-auth-request-email", uid) end
              if not h:get("x-goog-authenticated-user-email") then
                h:replace("x-goog-authenticated-user-email", "accounts.google.com:" .. uid)
              end
            end
YAML

echo "[kfp] disable app auth + unify user headers ..."
kubectl -n "${NS_KF}" set env deployment/centraldashboard                APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX= || true
kubectl -n "${NS_KF}" set env deployment/jupyter-web-app-deployment      APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX= || true
kubectl -n "${NS_KF}" set env deployment/tensorboards-web-app-deployment APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX= || true
kubectl -n "${NS_KF}" set env deployment/volumes-web-app-deployment      APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX= || true
kubectl -n "${NS_KF}" set env deployment/kserve-models-web-app           APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX= || true
kubectl -n "${NS_KF}" set env deployment/ml-pipeline-ui                  APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX= DISABLE_GKE_METADATA=true || true
kubectl -n "${NS_KF}" set env deployment/ml-pipeline                     KUBEFLOW_USERID_HEADER=kubeflow-userid KUBEFLOW_USERID_PREFIX= || true
# watch all the namespace
kubectl -n "${NS_KF}" set env deployment/ml-pipeline-persistenceagent    NAMESPACE= || true

echo "[kfp] mysql: disable istio sidecar + restart ..."
kubectl -n "${NS_KF}" annotate deploy/mysql sidecar.istio.io/inject="false" --overwrite || true
kubectl -n "${NS_KF}" rollout restart deploy/mysql || true
kubectl -n "${NS_KF}" rollout status  deploy/mysql --timeout=300s || true

echo "[kfp] RBAC mode = ${RBAC_MODE} ..."
if [ "${RBAC_MODE}" = "A" ]; then
  # all pass: cluster-admin（test env）
  kubectl create clusterrolebinding kfp-user-admin \
    --clusterrole=cluster-admin --user="${USER_EMAIL}" 2>/dev/null || true
else
  # Least Privilege: Only allow all verbs for pipelines.kubeflow.org/workflows
  kubectl delete clusterrolebinding kfp-user-admin --ignore-not-found
  cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kfp-workflows-any
rules:
- apiGroups: ["pipelines.kubeflow.org"]
  resources: ["workflows"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kfp-workflows-any-$(echo "${USER_EMAIL}" | tr '@' '-' | tr '/' '-')
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kfp-workflows-any
subjects:
- kind: User
  name: ${USER_EMAIL}
EOF
fi

echo "[kfp] ensure MinIO default bucket (mlpipeline) exists ..."
kubectl -n "${NS_KF}" exec deploy/minio -- sh -c '
mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" --api s3v4 &&
mc mb -p local/mlpipeline || true
' || true

echo "[kfp] restart KFP core to pick up changes ..."
for d in ml-pipeline ml-pipeline-persistenceagent ml-pipeline-ui; do
  kubectl -n "${NS_KF}" rollout restart deploy/$d || true
  kubectl -n "${NS_KF}" rollout status  deploy/$d --timeout=300s || true
done

echo "[kfp] optional: scale down oauth2-proxy / dex to avoid interference ..."
kubectl -n oauth2-proxy scale deploy -l app.kubernetes.io/name=oauth2-proxy --replicas=0 || true
kubectl -n auth         scale deploy -l app=dex                         --replicas=0 || true

echo "[kfp] done. UI should now reflect real run states."
