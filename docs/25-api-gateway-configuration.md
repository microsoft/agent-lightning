# API Gateway Configuration

Start the API Gateway with `agl-server`. It uses Hydra configuration, and the complete default configuration is located at `agentlightning/config/server.yaml`:

```yaml
host: 0.0.0.0
port: 8080
key: ""
default_proxy:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  include_log_probs: true
  include_routed_experts: false
  train:
    temperature: 1
  val:
    temperature: 0.7
```

## Override configuration

Override any setting with a Hydra command-line argument when starting the server. For example:

```bash
agl-server \
  host=0.0.0.0 \
  port=8080 \
  key="$AGL_KEY" \
  default_proxy.model_name=Qwen/Qwen3-8B
```

## Top-level settings

| Key | Default | Description |
|---|---:|---|
| `host` | `0.0.0.0` | Uvicorn bind address. |
| `port` | `8080` | Uvicorn listen port. |
| `key` | `""` | Bearer key for API and proxy routes. Empty disables authentication and logs a warning. |

Use the same non-empty key in the trainer and Controller.

## Proxy settings

| Key | Default | Description |
|---|---:|---|
| `default_proxy.model_name` | `Qwen/Qwen2.5-7B-Instruct` | Registered model name selected for forwarded requests. |
| `default_proxy.include_log_probs` | `true` | Ask the train backend for chosen-token log probabilities and token IDs. |
| `default_proxy.include_routed_experts` | `false` | Ask the train backend for routed-expert tensors, required for MoE routing replay (R3). |
| `default_proxy.train.temperature` | `1` | Temperature forced for training rollouts. |
| `default_proxy.val.temperature` | `0.7` | Temperature forced for validation rollouts. |

`default_proxy.model_name` must match the model configured in `verl` at `actor_rollout_ref.model.path`:

```text
server default_proxy.model_name
          = trainer actor_rollout_ref.model.path
```

A mismatch produces a “model not found” error even if the vLLM endpoint itself is healthy.

The train and validation temperatures configured here are the values actually used for model requests. Note that `verl` has similar temperature settings, but those values are not used for proxied requests because the proxy replaces them automatically.

We recommend keeping `default_proxy.include_log_probs: true`. This records rollout log probabilities and allows `verl` to report rollout-correction metrics. Some rollout-correction features also require these log probabilities.

For MoE routing replay, set `default_proxy.include_routed_experts: true` and use a backend that supports `return_routed_experts`. The gateway stores the returned tensor in the `model_request` event's `routed_experts` field and removes it from the response forwarded to the agent. This setting adds the request parameter only for training rollouts; validation requests are forwarded without that override.

## Client disconnects

When an agent disconnects during a model request, the gateway cancels its upstream request and any pending retry, then releases the request's in-flight slot. An interrupted request does not produce a successful model-request event. Requests that finish before the disconnect is observed retain their recorded result.

Pausing the gateway still allows connected agents' in-flight requests to finish normally before a model update.
