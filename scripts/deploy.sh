#!/bin/bash
# Deploy agl-lite infrastructure to K8s.
#
# Usage:
#   scripts/deploy.sh --agl-in-k8s       # agl-lite server + controller in K8s (default)
#   scripts/deploy.sh --agl-in-host      # controller in K8s + agl-lite server on host
#   scripts/deploy.sh --cleanup          # remove K8s namespace (and stop host server if started by this script)
#
# Modes:
#   --agl-in-k8s (default):
#     Both agl-lite server and controller run inside K8s.
#     AGL_LITE_URL is auto-set to http://agl-lite.<namespace>.svc:8080.
#
#   --agl-in-host:
#     Controller runs in K8s. agl-lite server is launched by this script on host.
#     AGL_LITE_URL defaults to http://host.minikube.internal:8080 on minikube.
#     On minikube, this script patches CoreDNS so pods can resolve host.minikube.internal.
#
# Backward compatibility:
#   --controller-only / --no-serve are aliases of --agl-in-host.
#
# Reads: deploy/.env (config), AGL_KEY env var or from .env (secret).
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
AGL_LOCATION="k8s" # k8s | host
for arg in "$@"; do
    case "$arg" in
        --agl-in-k8s) AGL_LOCATION="k8s" ;;
        --agl-in-host) AGL_LOCATION="host" ;;
        --controller-only) AGL_LOCATION="host" ;;
        # hidden backwards compat alias
        --no-serve) AGL_LOCATION="host" ;;
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

# --- Determine AGL_LITE_URL ---
if [ "$AGL_LOCATION" = "host" ]; then
    # Server runs on host (launched below).
    if [ -z "${AGL_LITE_URL:-}" ]; then
        ctx=$(kubectl config current-context 2>/dev/null || echo "")
        if [[ "$ctx" == "minikube" ]]; then
            AGL_LITE_URL="http://host.minikube.internal:8080"
        else
            AGL_LITE_URL="http://127.0.0.1:8080"
        fi
    fi
    echo "=== Mode: agl-in-host (server at $AGL_LITE_URL) ==="
else
    # Server runs in-cluster — auto-set URL to cluster-internal service DNS
    AGL_LITE_URL="http://agl-lite.${NS}.svc:8080"
    echo "=== Mode: agl-in-k8s (server at $AGL_LITE_URL) ==="
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

# 3. ConfigMap (from .env, excluding AGL_KEY and comments, with correct AGL_LITE_URL)
echo "--- Creating configmap ---"
(grep -v '^AGL_KEY=' "$ENV_FILE" | grep -v '^AGL_LITE_URL=' | grep -v '^#' | grep -v '^$'; \
 echo "AGL_LITE_URL=$AGL_LITE_URL") | \
    kubectl -n "$NS" create configmap agl-lite-config \
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
    echo "--- Skipping agl-lite server deployment in K8s (host mode) ---"
fi

echo "--- Deploying controller ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/k8s.yaml"

# 6. Minikube connectivity fix (host mode only)
if [ "$AGL_LOCATION" = "host" ] && echo "$AGL_LITE_URL" | grep -q "host.minikube.internal"; then
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

    HOST_PORT=$(echo "$AGL_LITE_URL" | sed -E 's#^https?://[^:/]+:([0-9]+).*$#\1#')
    if ! [[ "$HOST_PORT" =~ ^[0-9]+$ ]]; then
        HOST_PORT=8080
    fi

    SERVE_ARGS=(serve --host 0.0.0.0 --port "$HOST_PORT")
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
    echo "Host server pid: $(cat "$HOST_PID_FILE")"
    echo "Host server log: $HOST_LOG_FILE"
    echo "To stop host server: kill $(cat "$HOST_PID_FILE") && rm -f $HOST_PID_FILE"
else
    echo "Server is in-cluster at: $AGL_LITE_URL"
    echo "To access from host (for rl_loop.py or debugging):"
    echo "  kubectl -n $NS port-forward svc/agl-lite 8080:8080"
    echo "  export AGL_LITE_URL=http://localhost:8080 AGL_KEY=\$AGL_KEY"
fi
