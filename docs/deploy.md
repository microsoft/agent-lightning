# Deploy agl-lite

This guide describes the YAML-driven deploy command:

```bash
export AGL_KEY=$(openssl rand -hex 32)
agl-lite deploy --config deploy/agl-lite.yaml

# cleanup (delete namespace + stop managed host server if any)
agl-lite deploy --config deploy/agl-lite.yaml --cleanup
```

> `scripts/deploy.sh` is a thin wrapper around the same command.

---

## Deploy Process Overview

agl-lite supports three deployment modes. In all modes, the controller runs in Kubernetes.

| Mode | agl-lite server (gateway + store) location | Required fields |
|---|---|---|
| `agl-in-k8s` | In Kubernetes (`deploy/agl-lite/k8s.yaml`) | none |
| `agl-in-host` | On deploy host (`agl-lite serve`) | `agl_host_ip_bind`, `agl_host_port`; `agl_base_url_k8s_accessible` required on non-minikube |
| `agl-external` | External service (not managed by deploy) | `agl_base_url_k8s_accessible` |

The deploy flow:

```mermaid
flowchart TB
    A([Start]) --> M{mode}

    subgraph Host
      M -- agl-in-host --> H[Start agl-lite serve]
      H --> X[Resolve k8s-accessible URL]
      M -- agl-external --> X2[Use agl_base_url_k8s_accessible]
      M -- agl-in-k8s --> SKIP[Skip host steps]
    end

    subgraph K8s
      X --> P[K8s prepare: namespace / secret / configmap / RBAC]
      X2 --> P
      SKIP --> P
      P --> M2{mode}
      M2 -- agl-in-k8s --> AK[Deploy agl-lite service]
      AK --> AC[Deploy controller]
      M2 -- agl-in-host / agl-external --> AC
    end

    AC --> S([Succeeded])
```

---

## URL Model: pod-facing vs host-facing

Two URLs are used after deploy:

- **Pod-facing URL**: used by Kubernetes workloads (controller, and URLs injected into agent pods).
- **Host-facing URL**: used by host-side scripts/algorithms.

Deploy writes both to `${local_state_dir}/agl-lite.env` (default `.local/agl-lite.env`):

- `AGL_BASE_URL` = host-facing URL
- `AGL_BASE_URL_POD` = pod-facing URL

Mode behavior:

| Mode | Pod-facing URL (`AGL_BASE_URL` in K8s ConfigMap) | Host-facing URL (`AGL_BASE_URL` in `.local/agl-lite.env`) |
|---|---|---|
| `agl-in-k8s` | Auto: `http://agl-lite.<namespace>.svc:8080` | `http://127.0.0.1:<agl_host_port>` (requires manual `kubectl port-forward`) |
| `agl-in-host` | `agl_base_url_k8s_accessible`, or auto `http://host.minikube.internal:<agl_host_port>` on minikube | `http://127.0.0.1:<agl_host_port>` |
| `agl-external` | `agl_base_url_k8s_accessible` | `agl_base_url_k8s_accessible` |

Example for `agl-in-k8s` host access:

```bash
kubectl -n <namespace> port-forward svc/agl-lite 8080:8080
# then host-side AGL_BASE_URL can use http://127.0.0.1:8080
```

---

## Config

Start from `deploy/agl-lite.yaml.example`.

```yaml
namespace: agl
mode: agl-in-k8s

# required for agl-external; required for agl-in-host on non-minikube
# agl_base_url_k8s_accessible: http://my-agl-lite.example:8080

agl_host_ip_bind: 0.0.0.0
agl_host_port: 8080

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

Field summary:

| Field | Default | Description |
|---|---:|---|
| `namespace` | — | Kubernetes namespace to deploy into |
| `mode` | — | `agl-in-k8s` / `agl-in-host` / `agl-external` |
| `agl_base_url_k8s_accessible` | `null` | URL reachable from pods |
| `agl_host_ip_bind` | `0.0.0.0` | bind IP for host `agl-lite serve` (host mode) |
| `agl_host_port` | `8080` | host serve port and local host-facing URL port |
| `controller.poll_interval_seconds` | `10` | controller poll interval |
| `controller.max_queue_time_seconds` | `3600` | max queue time before timeout |
| `server_runtime.gateway_config` | `null` | optional gateway config path |
| `server_runtime.hooks` | `null` | optional hooks module path |
| `server_runtime.artifact_dir` | `null` | optional artifact directory |
| `wait_ready_timeout_seconds` | `120` | readiness wait timeout |
| `local_state_dir` | `.local` | location of generated env/pid/log files |

Validation rules:

- `mode=agl-in-k8s`:
  - `agl_base_url_k8s_accessible` must be unset.
- `mode=agl-in-host`:
  - starts host `agl-lite serve`.
  - on non-minikube, `agl_base_url_k8s_accessible` is required.
  - on minikube, pod URL defaults to `http://host.minikube.internal:<agl_host_port>`.
- `mode=agl-external`:
  - `agl_base_url_k8s_accessible` is required and used for both pod + host URLs.

---

## Outputs

After successful deploy:

- Kubernetes resources are applied in `<namespace>`.
- `.local/agl-lite.env` (or `${local_state_dir}/agl-lite.env`) is generated.
- In `agl-in-host`, host server PID/log are written under `${local_state_dir}`:
  - `agl-lite-serve.pid`
  - `agl-lite-serve.log`
