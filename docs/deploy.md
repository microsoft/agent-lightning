# Deploy agl-lite

This guide explains how to deploy agl-lite with the new YAML-based deploy command:

```bash
export AGL_KEY=$(openssl rand -hex 32)
agl-lite deploy --config deploy/agl-lite.yaml
```

> `scripts/deploy.sh` is a thin wrapper around the same command.

---

## Deploy Modes

To support different environments, agl-lite has three deployment modes:

| Mode           | Where agl-lite server runs                 | Use Case                                                    |
|---             |---                                         |---                                                          |
| `agl-in-k8s`   | In Kubernetes (`deploy/agl-lite/k8s.yaml`) | Typical Kubernetes clusters (dev and prod).                 |
| `agl-in-host`  | On deploy host (`agl-lite serve`)          | agl-lite located in the computing backend, e.g., GPU nodes. |
| `agl-external` | External service (not managed)             | Managed services, or when deployment is handled separately. |

> The field `mode` in the deploy config determines the deployment mode. See [Config file](#3-config-file) for details.

### URL model: pod-facing vs host-facing

After deployment, the agl-lite server must be reachable from both:
- **Controller and agent pods** inside Kubernetes (pod-facing).
- **Host-side scripts (algorithm)** on the deploy host (host-facing), particularly for computing backends like GPU nodes that run host-side algorithm scripts.

Thus there are **two URL perspectives** in agl-lite deployments:

- **Pod-facing URL**: used by controller and agent pods inside Kubernetes.
- **Host-facing URL**: used by host-side algorithm scripts, debugging tools, and CLI.

`agl-lite deploy` computes both and writes them to:

- `${local_state_dir}/agl-lite.env` (default: `.local/agl-lite.env`)

That file exports:

- `AGL_BASE_URL` (host-facing)
- `AGL_BASE_URL_POD` (pod-facing)
- `AGL_K8S_NAMESPACE`

Use it after deploy:

```bash
source .local/agl-lite.env
```

---

## 2) Deployment modes and URL behavior

| Mode           | Where agl-lite server runs                 | Pod-facing URL (`AGL_BASE_URL_POD`)                                                 | Host-facing URL (`AGL_BASE_URL`)                          |
|---             |---                                         |---                                                                                  |---                                                        |
| `agl-in-k8s`   | In Kubernetes (`deploy/agl-lite/k8s.yaml`) | Auto: `http://agl-lite.<namespace>.svc:8080`                                        | `http://127.0.0.1:8080` (requires `kubectl port-forward`) |
| `agl-in-host`  | On deploy host (`agl-lite serve`)          | `agl_base_url_pod` from config, or auto `host.minikube.internal:<port>` on minikube | `http://127.0.0.1:<host_serve.port>`                      |
| `agl-external` | External service (not managed)             | `agl_base_url_external`                                                             | `agl_base_url_external`                                   |

### Important

- In `agl-in-k8s`, host URL only works after:
  ```bash
  kubectl -n <namespace> port-forward svc/agl-lite 8080:8080
  ```
- In remote clusters, pod-facing URL must be reachable from pods (no `localhost` / `127.0.0.1`).

---

## 3) Config file

Start from:

- `deploy/agl-lite.yaml.example`

Typical config:

```yaml
namespace: agl
mode: agl-in-k8s

# agl_base_url_pod: http://host.minikube.internal:8080   # for agl-in-host (optional on minikube)
# agl_base_url_external: http://external-host:8080       # required for agl-external

host_serve:
  bind: 0.0.0.0
  port: 8080

controller:
  poll_interval_seconds: 10
  max_queue_time_seconds: 3600

server_runtime:
  gateway_config: null
  hooks: null
  artifact_dir: null

wait_ready_timeout_seconds: 120
local_state_dir: .local
```

### Field summary

- `namespace`: K8s namespace for agl-lite/controller resources.
- `mode`: one of `agl-in-k8s`, `agl-in-host`, `agl-external`.
- `agl_base_url_pod`: pod-facing URL override (host mode scenarios).
- `agl_base_url_external`: external base URL (external mode only).
- `host_serve.*`: host server bind/port (used in host mode).
- `controller.*`: controller reconcile behavior.
- `server_runtime.*`: optional runtime paths for gateway/hooks/artifacts.
- `wait_ready_timeout_seconds`: readiness timeout.
- `local_state_dir`: directory for generated `.env`, PID, and host log files.

---

## 4) Deploy workflow

### Deploy

```bash
export AGL_KEY=$(openssl rand -hex 32)
agl-lite deploy --config deploy/agl-lite.yaml
```

### Use generated env

```bash
source .local/agl-lite.env
echo "$AGL_BASE_URL"
```

### Cleanup

```bash
agl-lite deploy --config deploy/agl-lite.yaml --cleanup
```

Cleanup removes the namespace and stops host-managed `agl-lite serve` if present.

---

## 5) Examples by mode

### A) In-K8s mode

```yaml
namespace: agl
mode: agl-in-k8s
local_state_dir: .local
```

Then on host:

```bash
kubectl -n agl port-forward svc/agl-lite 8080:8080
source .local/agl-lite.env
```

### B) In-Host mode (minikube)

```yaml
namespace: agl
mode: agl-in-host
host_serve:
  bind: 0.0.0.0
  port: 8080
local_state_dir: .local
```

`agl_base_url_pod` can be omitted on minikube (auto uses `host.minikube.internal`).

### C) External mode

```yaml
namespace: agl
mode: agl-external
agl_base_url_external: http://external-agl-lite.example.com:8080
local_state_dir: .local
```

---

## 6) Validation and common errors

The deploy config is schema-validated. Typical failures:

- `AGL_KEY` missing in environment.
- `agl_base_url_external` missing when `mode=agl-external`.
- `agl_base_url_pod` set in disallowed mode.
- pod-facing URL points to localhost on non-minikube cluster.
- In host mode, URL port mismatch with `host_serve.port`.

Fix config and rerun deploy.
