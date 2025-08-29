#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] line:$LINENO cmd:$BASH_COMMAND"' ERR

# ===== 0) 变量与校验 =====
export NS_ISTIO="${NS_ISTIO:-istio-system}"
export NS_KF="${NS_KF:-kubeflow}"

: "${NS_ISTIO:?please export NS_ISTIO (e.g., istio-system)}"
: "${NS_KF:?please export NS_KF (e.g., kubeflow)}"

echo "[INFO] NS_ISTIO=$NS_ISTIO, NS_KF=$NS_KF"

# （0.5）确保 kubeflow 命名空间开启了 Istio 自动注入
inj="$(kubectl get ns "$NS_KF" -o jsonpath='{.metadata.labels.istio-injection}' || true)"
if [[ "$inj" != "enabled" ]]; then
  echo "[STEP] Enable istio-injection on namespace $NS_KF ..."
  kubectl label ns "$NS_KF" istio-injection=enabled --overwrite
fi

# ===== 1) IngressGateway -> NodePort（幂等）=====
echo "[STEP] Patch istio-ingressgateway Service to NodePort ..."
kubectl --namespace "$NS_ISTIO" patch svc istio-ingressgateway -p '{
  "spec": {
    "type": "NodePort",
    "ports": [
      {"name":"http2","port":80,"targetPort":8080,"nodePort":30080},
      {"name":"https","port":443,"targetPort":8443,"nodePort":30443}
    ]
  }
}' || true
kubectl --namespace "$NS_ISTIO" get svc istio-ingressgateway -o wide

# ===== 2) 清理 Istio 鉴权/认证（只清本命名空间与全局常见残留）=====
echo "[STEP] Delete RequestAuthentication in $NS_ISTIO ..."
kubectl --namespace "$NS_ISTIO" get requestauthentication -o name \
  | xargs -r -L1 kubectl --namespace "$NS_ISTIO" delete

echo "[STEP] Delete possible EnvoyFilters (oauth2-proxy/authn) in $NS_ISTIO ..."
for n in authn-filter jwt-authn oauth2-proxy; do
  kubectl --namespace "$NS_ISTIO" delete envoyfilter "$n" --ignore-not-found
done

echo "[STEP] Delete AuthorizationPolicy in $NS_ISTIO and $NS_KF ..."
kubectl --namespace "$NS_ISTIO" get authorizationpolicy -o name \
  | xargs -r -L1 kubectl --namespace "$NS_ISTIO" delete
kubectl --namespace "$NS_KF" get authorizationpolicy -o name \
  | xargs -r -L1 kubectl --namespace "$NS_KF" delete

echo "[INFO] List cluster AuthorizationPolicies (should be empty or minimal):"
kubectl get authorizationpolicy -A || true

# ===== 3) 在 kubeflow 放“全放行”兜底策略 =====
echo "[STEP] Apply allow-all AuthorizationPolicy in $NS_KF ..."
cat > allow-all-kubeflow.yaml <<'YAML'
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: allow-all-kubeflow
  namespace: kubeflow
spec:
  rules:
    - {}
YAML
kubectl apply -f allow-all-kubeflow.yaml

# ===== 4) 创建/更新固定用户 Profile =====
echo "[STEP] Ensure fixed user Profile exists ..."
cat > profile-user-example.yaml <<'YAML'
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: kubeflow-user-example-com
  namespace: kubeflow
spec:
  owner:
    kind: User
    name: user@example.com
YAML
kubectl apply -f profile-user-example.yaml

# ===== 5) 网关层强制注入固定用户请求头，并仅在网关移除 Authorization =====
echo "[STEP] Apply EnvoyFilter at gateway to force user headers ..."
cat > force-kubeflow-user.yaml <<'YAML'
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: force-kubeflow-user
  namespace: istio-system
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
              local uid = "user@example.com"
              local h = handle:headers()
              -- 统一伪装为 uid
              h:replace("kubeflow-userid", uid)
              h:replace("x-auth-request-email", uid)
              -- KFP 常优先读它，且很多环境要求加 'accounts.google.com:' 前缀
              h:replace("x-goog-authenticated-user-email", "accounts.google.com:" .. uid)
              -- 只在网关剥离 Authorization
              h:remove("authorization")
            end
YAML
kubectl apply -f force-kubeflow-user.yaml
kubectl --namespace "$NS_ISTIO" rollout status deploy/istio-ingressgateway

# ===== 6) 仅对 ml-pipeline 入站兜底补头（不要在这里删除 Authorization）=====
echo "[STEP] Apply namespace inbound EnvoyFilter fallback (ml-pipeline only) ..."
cat > add-userid-to-ml-pipeline-inbound.yaml <<'YAML'
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: add-userid-to-ml-pipeline-inbound
  namespace: kubeflow
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
              local h = handle:headers()
              local uid = "user@example.com"
              if not h:get("kubeflow-userid") then
                h:replace("kubeflow-userid", uid)
              end
              if not h:get("x-auth-request-email") then
                h:replace("x-auth-request-email", uid)
              end
              if not h:get("x-goog-authenticated-user-email") then
                h:replace("x-goog-authenticated-user-email", "accounts.google.com:" .. uid)
              end
              -- 注意：这里不再移除 Authorization
            end
YAML
kubectl apply -f add-userid-to-ml-pipeline-inbound.yaml

# （6.5）确保 ml-pipeline / ml-pipeline-ui 实际带有 sidecar；若无则重建
echo "[STEP] Ensure sidecar injected for ml-pipeline / ml-pipeline-ui ..."
if ! kubectl -n "$NS_KF" get pod -l app=ml-pipeline -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.name}{" "}{end}' | grep -qw istio-proxy; then
  kubectl -n "$NS_KF" delete pod -l app=ml-pipeline || true
fi
if ! kubectl -n "$NS_KF" get pod -l app=ml-pipeline-ui -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.name}{" "}{end}' | grep -qw istio-proxy; then
  kubectl -n "$NS_KF" delete pod -l app=ml-pipeline-ui || true
fi

# ===== 7) 关闭 WebApps 自身鉴权，并指定读取的头（WebApp 使用 USERID_* 变量）=====
echo "[STEP] Disable per-app auth and set USERID_HEADER=kubeflow-userid ..."
kubectl --namespace "$NS_KF" set env deployment/centraldashboard                     APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX=
kubectl --namespace "$NS_KF" set env deployment/jupyter-web-app-deployment           APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX=
kubectl --namespace "$NS_KF" set env deployment/tensorboards-web-app-deployment      APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX=
kubectl --namespace "$NS_KF" set env deployment/volumes-web-app-deployment           APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX=
kubectl --namespace "$NS_KF" set env deployment/kserve-models-web-app                APP_DISABLE_AUTH=True USERID_HEADER=kubeflow-userid USERID_PREFIX=

# （7b）关键：给 KFP 后端设置它自己的变量名（KUBEFLOW_USERID_*）
echo "[STEP] Set KFP backend env (KUBEFLOW_USERID_*) ..."
kubectl --namespace "$NS_KF" set env deployment/ml-pipeline \
  KUBEFLOW_USERID_HEADER=kubeflow-userid \
  KUBEFLOW_USERID_PREFIX=

echo "[STEP] Wait rollout of key components ..."
for d in centraldashboard jupyter-web-app-deployment tensorboards-web-app-deployment volumes-web-app-deployment kserve-models-web-app ml-pipeline; do
  kubectl --namespace "$NS_KF" rollout status deploy/$d
done

# ===== 8)（可选）减少干扰：缩容 oauth2-proxy / Dex =====
echo "[STEP] (Optional) scale down oauth2-proxy and dex to avoid side effects ..."
kubectl --namespace oauth2-proxy scale deploy -l app.kubernetes.io/name=oauth2-proxy --replicas=0 || true
kubectl --namespace auth         scale deploy -l app=dex                         --replicas=0 || true

echo "[DONE] Auth removed; all traffic to Kubeflow will be treated as user@example.com"
echo "[HINT] Access via:  http://<node-ip>:30080/"
