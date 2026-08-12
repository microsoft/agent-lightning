# Controller Configuration

The Controller translates declarative rollouts into real agent executions. It uses Hydra configuration, and the complete default configuration is located at `agentlightning/config/controller.yaml`:

![Controller reconciliation](images/controller-reconciliation.jpg)

```yaml
runner_type: k8s

agl_server:
  url: http://localhost:8080
  agent_url: null
  key: ""

k8s_runner:
  namespace: default
  ttl_after_finished: 1200
  max_jobs_per_minute: 100
  poll_interval: 5

local_runner:
  maximum_size: 50
  poll_interval: 10
```

## Override configuration

Override any setting with a Hydra command-line argument when starting the Controller. For example:

```bash
agl-controller \
  runner_type=local \
  agl_server.url=http://localhost:8080 \
  agl_server.key="$AGL_KEY" \
  local_runner.maximum_size=32
```
## Runner type

The Controller supports one runner type at a time. Set `runner_type` to either `k8s` or `local`:

| Key | Default | Description |
|---|---:|---|
| `runner_type` | `k8s` | The single execution backend used by this Controller: `k8s` or `local`. |

One Controller instance cannot run both modes simultaneously.

In `k8s` mode, every rollout runs as a Kubernetes Job. The Controller uses the default Kubernetes configuration on its machine to access the cluster. In `local` mode, every rollout runs as a local subprocess on the Controller machine, with multiple rollouts managed through a local process pool.

## Connect to the server

The Controller configuration contains two Gateway URLs for two different network paths:

| Key | Default | Description |
|---|---:|---|
| `agl_server.url` | `http://localhost:8080` | URL used by the Controller itself to request the Agent Lightning Gateway server. |
| `agl_server.agent_url` | `null` | URL used by the Agent to request the Agent Lightning Gateway server. When `null`, it falls back to `agl_server.url`. |
| `agl_server.key` | `""` | Bearer key used by the Controller and agents. |

`agl_server.url` must be reachable from the Controller process. `agl_server.agent_url` must be reachable from the Agent process or pod because it is used to build the Gateway proxy and event URLs injected into that Agent.

In most cases, `agl_server.agent_url` does not need to be set separately. Leave it as `null`, and Agents automatically use `agl_server.url` to access the Gateway server.

Set `agl_server.agent_url` only when Agents cannot reach the Gateway through `agl_server.url`, usually because the Controller and Agents are in different networks. For example, with the Minikube Docker driver, the Controller may use `http://localhost:8080`, while Agents inside Minikube need `http://host.minikube.internal:8080` to access the same Gateway server.

## K8s runner limits

The K8s runner provides settings that limit Job creation and clean up completed Jobs:

| Key | Default | Description |
|---|---:|---|
| `k8s_runner.max_jobs_per_minute` | `100` | Maximum number of Kubernetes Jobs the Controller can create per minute. |
| `k8s_runner.ttl_after_finished` | `1200` | Number of seconds a completed Job is retained before Kubernetes removes it automatically. |

`max_jobs_per_minute` prevents the Controller from creating too many Jobs in a short period. `ttl_after_finished` prevents completed Jobs from accumulating and overloading the Kubernetes API server.

## Local runner limits

The local runner limits concurrent processes and periodically synchronizes their state:

| Key | Default | Description |
|---|---:|---|
| `local_runner.maximum_size` | `50` | Maximum number of Agent subprocesses managed concurrently on the Controller machine. |
| `local_runner.poll_interval` | `10` | Number of seconds between automatic synchronization checks for local process and rollout state. |

When the process pool reaches `maximum_size`, queued rollouts wait until capacity becomes available. A lower `poll_interval` detects process-state changes sooner but performs synchronization more frequently.

## How tasks are launched

In `k8s` mode, the Controller reads the Jinja Job template stored in each rollout. The template originates from `agentlightning.k8s.job_template_path` in the Trainer configuration. The Controller renders the template with the rollout's `input`, applies the Controller settings such as namespace, timeout, and cleanup TTL, and submits the resulting Kubernetes Job. It also injects rollout-specific `AGL_OPENAI_BASE_URL`, `AGL_EVENT_URL`, and `AGL_KEY` values into every container. See [Trainer Configuration](4-trainer-configuration.md#rollout-execution) for the template configuration and Jinja examples, and `agentlightning/controller/k8s_reconciler.py` for the implementation.

In `local` mode, the Controller reads `agent_class` and `env_map` from the rollout. It imports the Agent class, starts it in a local subprocess, and uses `env_map` to replace environment-variable values with fields from the rollout's `input`. The same rollout-specific Gateway URL, event URL, and key are injected automatically.

