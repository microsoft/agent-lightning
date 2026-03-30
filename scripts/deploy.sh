#!/bin/bash
# Deploy agl-lite infrastructure to K8s.
#
# Usage:
#   scripts/deploy.sh --agl-in-k8s                     # agl-lite server + controller in K8s (default)
#   scripts/deploy.sh --agl-in-host [--agl-host-port N --agl-host-bind IP]
#                                                      # controller in K8s + agl-lite server on host
#   scripts/deploy.sh --agl-external                  # controller in K8s; agl-lite already running elsewhere
#   scripts/deploy.sh --cleanup                       # remove K8s namespace (and stop host server if started by this script)
#
# Modes:
#   --agl-in-k8s (default):
#     Both agl-lite server and controller run inside K8s.
#     AGL_LITE_URL is auto-set to http://agl-lite.<namespace>.svc:8080.
#
#   --agl-in-host:
#     Controller runs in K8s. agl-lite server is launched by this script on host.
#     On minikube, AGL_LITE_URL defaults to http://host.minikube.internal:<port>.
#     On non-minikube clusters, AGL_LITE_URL must be set explicitly to a pod-reachable host URL.
#
#   --agl-external:
#     Controller runs in K8s. agl-lite service is NOT launched by this script.
#     AGL_LITE_URL must be set explicitly to a pod-reachable URL.
#
# Backward compatibility:
#   --controller-only / --no-serve are aliases of --agl-in-host.
#
# Reads: deploy/.env (config), AGL_KEY env var or from .env (secret).
# 
# Outputs:
#   - AGL_LITE_URL (printed at end and set in configmap for controller) 
#     Both controller and agent pods read AGL_LITE_URL from configmap and use it to connect to the server.
# 
# Notes:
#   - This script presumes that kubectl context is set to the target cluster (e.g. minikube).
#   - This script handles network connectivity as follows:
#
# | agl location | k8s type | pod-facing URL (AGL_LITE_URL)        | server binding (agl-lite serve --host)                           | Auto set in deploy.sh            |
# |--------------|----------|----------------------------------------|-------------------------------------------------------------------|----------------------------------|
# | in-k8s       | any      | http://agl-lite.<ns>.svc:<port>       | N/A (k8s service handles routing)                                 | Yes                              |
# | external     | any      | http(s)://<external-host>:<port>      | N/A (not launched by this script)                                 | No (user must set AGL_LITE_URL) |
# | in-host      | minikube | http://host.minikube.internal:<port>  | 0.0.0.0 (recommended) or specific bind IP                         | Yes (with minikube DNS patch)    |
# | in-host      | remote   | http://<public-or-routable-ip>:<port> | 0.0.0.0 or specific bind IP (must be reachable from the cluster)  | No (user must set AGL_LITE_URL) |
#
#   - For host mode on minikube, this script patches CoreDNS so pods can resolve
#     host.minikube.internal to the minikube host IP.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/deploy/.env"

# --- Load config ---
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: deploy/.env not found. Copy from deploy/.env.example and edit."
    exit 1
fi
source "$ENV_FILE"

NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set in .env}"

HOST_STATE_DIR="$REPO_ROOT/.local"
HOST_PID_FILE="$HOST_STATE_DIR/agl-lite-serve.pid"
HOST_LOG_FILE="$HOST_STATE_DIR/agl-lite-serve.log"

stop_host_server_if_running() {
    if [ -f "$HOST_PID_FILE" ]; then
        local pid
        pid=$(cat "$HOST_PID_FILE" || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "--- Stopping host agl-lite server (pid=$pid) ---"
            kill "$pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$HOST_PID_FILE"
    fi
}

# --- Cleanup mode ---
if [[ "${1:-}" == "--cleanup" || "${1:-}" == "--teardown" ]]; then
    echo "=== Cleaning up namespace: $NS ==="
    kubectl delete namespace "$NS" --ignore-not-found --wait
    stop_host_server_if_running
    echo "Done."
    exit 0
fi

# --- Parse flags ---
AGL_LOCATION="k8s" # k8s | host | external
AGL_HOST_BIND="${AGL_HOST_BIND:-0.0.0.0}"
AGL_HOST_PORT="${AGL_HOST_PORT:-8080}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agl-in-k8s) AGL_LOCATION="k8s" ; shift ;;
        --agl-in-host) AGL_LOCATION="host" ; shift ;;
        --agl-external) AGL_LOCATION="external" ; shift ;;
        --controller-only) AGL_LOCATION="host" ; shift ;;
        --no-serve) AGL_LOCATION="host" ; shift ;;
        --agl-host-bind)
            AGL_HOST_BIND="$2"
            shift 2
            ;;
        --agl-host-port)
            AGL_HOST_PORT="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown flag: $1"
            exit 1
            ;;
    esac
done

# --- Helper: ensure host.minikube.internal resolves from pods ---
ensure_minikube_host_dns() {
    # Only relevant on minikube — check by context name
    local ctx
    ctx=$(kubectl config current-context 2>/dev/null || echo "")
    if [[ "$ctx" != "minikube" ]]; then
        return 0
    fi

    # Check if CoreDNS already has a hosts block for host.minikube.internal
    local corefile
    corefile=$(kubectl -n kube-system get configmap coredns -o jsonpath='{.data.Corefile}' 2>/dev/null || echo "")
    if echo "$corefile" | grep -q "host.minikube.internal"; then
        echo "✓ CoreDNS already resolves host.minikube.internal"
        return 0
    fi

    echo "⚠ Patching CoreDNS so pods can resolve host.minikube.internal..."
    local host_ip
    host_ip=$(minikube ssh "ip route | grep default | awk '{print \$3}'" 2>/dev/null | tr -d '\r\n')
    if [ -z "$host_ip" ]; then
        echo "WARNING: Could not detect minikube host IP. Pods may not reach the external server."
        return 1
    fi

    # Inject hosts block into Corefile (after the health check block)
    local new_corefile
    new_corefile=$(echo "$corefile" | sed "/^    ready$/a\\
    hosts {\\
       $host_ip host.minikube.internal\\
       fallthrough\\
    }")

    kubectl -n kube-system get configmap coredns -o json | \
        python3 -c "
import sys, json
cm = json.load(sys.stdin)
cm['data']['Corefile'] = '''$new_corefile'''
json.dump(cm, sys.stdout)
" | kubectl apply -f - >/dev/null

    # Restart CoreDNS to pick up changes
    kubectl -n kube-system rollout restart deployment coredns >/dev/null 2>&1
    kubectl -n kube-system wait --for=condition=available deployment/coredns --timeout=30s >/dev/null 2>&1
    echo "✓ CoreDNS patched: host.minikube.internal → $host_ip"
}

# --- Check AGL_KEY ---
if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Either:"
    echo "  1. export AGL_KEY=\$(openssl rand -hex 32)"
    echo "  2. Uncomment AGL_KEY in deploy/.env"
    exit 1
fi

# --- Redeploy warnings (non-blocking) ---
if kubectl -n "$NS" get deployment agl-controller >/dev/null 2>&1; then
    echo ""
    echo "⚠ Existing deployment detected in namespace '$NS' (agl-controller already exists)."
    echo "  Recommended for a clean restart:"
    echo "    scripts/deploy.sh --cleanup"
    echo "    scripts/deploy.sh --$([ "$AGL_LOCATION" = "k8s" ] && echo 'agl-in-k8s' || ([ "$AGL_LOCATION" = "host" ] && echo 'agl-in-host' || echo 'agl-external'))"
    echo ""
    echo "⚠ If deploying from source changes, rebuild images first (minikube):"
    echo "    scripts/build_images.sh"
    echo "  (for math-poc agents too: scripts/build_images.sh --math-poc)"
fi

# --- Determine AGL_LITE_URL ---
CTX=$(kubectl config current-context 2>/dev/null || echo "")

if [ "$AGL_LOCATION" = "k8s" ]; then
    # Server runs in-cluster — auto-set URL to cluster-internal service DNS
    AGL_LITE_URL="http://agl-lite.${NS}.svc:8080"
    echo "=== Mode: agl-in-k8s (server at $AGL_LITE_URL) ==="

elif [ "$AGL_LOCATION" = "host" ]; then
    # Server runs on host (launched below).
    if [ -z "${AGL_LITE_URL:-}" ]; then
        if [[ "$CTX" == "minikube" ]]; then
            AGL_LITE_URL="http://host.minikube.internal:${AGL_HOST_PORT}"
        else
            echo "ERROR: --agl-in-host on non-minikube requires explicit AGL_LITE_URL"
            echo "  Example: AGL_LITE_URL=http://<routable-host-ip>:${AGL_HOST_PORT}"
            exit 1
        fi
    fi

    if [[ "$CTX" != "minikube" ]] && echo "$AGL_LITE_URL" | grep -Eq '://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|/|$)'; then
        echo "ERROR: AGL_LITE_URL=$AGL_LITE_URL is not pod-reachable for remote clusters."
        echo "  Use a routable host IP or DNS name."
        exit 1
    fi

    echo "=== Mode: agl-in-host (server at $AGL_LITE_URL) ==="

else
    # external mode: must be explicitly provided by user
    if [ -z "${AGL_LITE_URL:-}" ]; then
        echo "ERROR: --agl-external requires AGL_LITE_URL in deploy/.env"
        echo "  Example: AGL_LITE_URL=http://<external-host>:8080"
        exit 1
    fi
    if echo "$AGL_LITE_URL" | grep -Eq '://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|/|$)'; then
        echo "ERROR: AGL_LITE_URL=$AGL_LITE_URL is not pod-reachable for --agl-external."
        exit 1
    fi
    echo "=== Mode: agl-external (server at $AGL_LITE_URL) ==="
fi

echo "=== Deploying to namespace: $NS ==="

# 1. Namespace
echo "--- Creating namespace ---"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# 2. Secret (AGL_KEY — from env var, never written to disk by this script)
echo "--- Creating secret ---"
kubectl -n "$NS" create secret generic "${AGL_SECRET_NAME:-agl-lite-keys}" \
    --from-literal=AGL_KEY="$AGL_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. ConfigMap (from .env + runtime overrides, excluding AGL_KEY)
echo "--- Creating configmap ---"
(
  grep -v '^AGL_KEY=' "$ENV_FILE" \
    | grep -v '^AGL_LITE_URL=' \
    | grep -v '^AGL_GATEWAY_CONFIG=' \
    | grep -v '^AGL_HOOKS=' \
    | grep -v '^AGL_ARTIFACT_DIR=' \
    | grep -v '^#' | grep -v '^$'
  echo "AGL_LITE_URL=$AGL_LITE_URL"
  if [ -n "${AGL_GATEWAY_CONFIG:-}" ]; then
    echo "AGL_GATEWAY_CONFIG=$AGL_GATEWAY_CONFIG"
  fi
  if [ -n "${AGL_HOOKS:-}" ]; then
    echo "AGL_HOOKS=$AGL_HOOKS"
  fi
  if [ -n "${AGL_ARTIFACT_DIR:-}" ]; then
    echo "AGL_ARTIFACT_DIR=$AGL_ARTIFACT_DIR"
  fi
) | kubectl -n "$NS" create configmap agl-lite-config \
    --from-env-file=/dev/stdin \
    --dry-run=client -o yaml | kubectl apply -f -

# 4. RBAC
echo "--- Applying RBAC ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/rbac.yaml"

# 5. Deployments
if [ "$AGL_LOCATION" = "k8s" ]; then
    echo "--- Deploying agl-lite server (K8s) ---"
    kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/agl-lite/k8s.yaml"
else
    echo "--- Skipping agl-lite server deployment in K8s ($AGL_LOCATION mode) ---"
fi

echo "--- Deploying controller ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/k8s.yaml"

# 6. Minikube connectivity fix (when using host.minikube.internal)
if [ "$AGL_LOCATION" != "k8s" ] && echo "$AGL_LITE_URL" | grep -q "host.minikube.internal"; then
    ensure_minikube_host_dns
fi

# 7. Wait
if [ "$AGL_LOCATION" = "k8s" ]; then
    echo "--- Waiting for pods ---"
    kubectl -n "$NS" wait --for=condition=available deployment/agl-lite --timeout=120s
    kubectl -n "$NS" wait --for=condition=available deployment/agl-controller --timeout=120s
else
    echo "--- Waiting for controller ---"
    kubectl -n "$NS" wait --for=condition=available deployment/agl-controller --timeout=120s
fi

# 8. Start host agl-lite server if needed
if [ "$AGL_LOCATION" = "host" ]; then
    echo "--- Launching agl-lite server on host ---"
    mkdir -p "$HOST_STATE_DIR"
    stop_host_server_if_running

    HOST_PORT="$AGL_HOST_PORT"
    if [[ "$AGL_LITE_URL" =~ :([0-9]+)$ ]]; then
        HOST_PORT="${BASH_REMATCH[1]}"
    fi

    SERVE_ARGS=(serve --host "$AGL_HOST_BIND" --port "$HOST_PORT")
    [ -n "${AGL_GATEWAY_CONFIG:-}" ] && SERVE_ARGS+=(--gateway-config "$AGL_GATEWAY_CONFIG")
    [ -n "${AGL_HOOKS:-}" ] && SERVE_ARGS+=(--hooks "$AGL_HOOKS")
    [ -n "${AGL_ARTIFACT_DIR:-}" ] && SERVE_ARGS+=(--artifact-dir "$AGL_ARTIFACT_DIR")

    nohup env AGL_KEY="$AGL_KEY" uv run agl-lite "${SERVE_ARGS[@]}" > "$HOST_LOG_FILE" 2>&1 &
    HOST_PID=$!
    echo "$HOST_PID" > "$HOST_PID_FILE"

    # Health check from host-side URL (replace host.minikube.internal with localhost)
    HEALTH_URL="$AGL_LITE_URL"
    HEALTH_URL=${HEALTH_URL/host.minikube.internal/localhost}
    for i in $(seq 1 40); do
        if curl -sf "$HEALTH_URL/healthz" >/dev/null 2>&1; then
            echo "✓ Host agl-lite server ready (pid=$HOST_PID)"
            break
        fi
        if ! kill -0 "$HOST_PID" 2>/dev/null; then
            echo "ERROR: host agl-lite server exited; see $HOST_LOG_FILE"
            tail -40 "$HOST_LOG_FILE" || true
            exit 1
        fi
        sleep 1
    done
fi

echo ""
echo "=== agl-lite deployed to namespace: $NS ==="
kubectl -n "$NS" get pods
echo ""
if [ "$AGL_LOCATION" = "host" ]; then
    echo "Server is running on host at: $AGL_LITE_URL"
    echo "Host bind/port: $AGL_HOST_BIND:$HOST_PORT"
    echo "Host server pid: $(cat "$HOST_PID_FILE")"
    echo "Host server log: $HOST_LOG_FILE"
    echo "To stop host server: kill $(cat "$HOST_PID_FILE") && rm -f $HOST_PID_FILE"
elif [ "$AGL_LOCATION" = "external" ]; then
    echo "Server is external (not managed by this script): $AGL_LITE_URL"
else
    echo "Server is in-cluster at: $AGL_LITE_URL"
    echo "To access from host (for rl_loop.py or debugging):"
    echo "  kubectl -n $NS port-forward svc/agl-lite 8080:8080"
    echo "  export AGL_LITE_URL=http://localhost:8080 AGL_KEY=\$AGL_KEY"
fi
