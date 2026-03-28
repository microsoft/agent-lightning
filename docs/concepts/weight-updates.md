# Weight Updates

agl-lite coordinates training and inference through standard CRUD operations on the model server registry. There is no special "update mode" or "pause" API — the gateway's behavior emerges naturally from whether servers are registered.

## The protocol

### Synchronous RL (stop-the-world)

```
Trainer                                    agl-lite Gateway
───────                                    ────────────────
1. Training step complete

2. DELETE /api/models/qwen-7b        ──→   Pool empty → new requests get 503 + Retry-After
                                           (in-flight requests complete normally)

3. Kill old vLLM, launch new with updated weights, wait for ready

4. POST /api/models                  ──→   Servers registered (version: 43)
   [{model: "qwen-7b",                     Routing resumes
     endpoint: "...", version: 43}]         Retrying agents succeed transparently
```

### Online RL (rolling update — no downtime)

```
1. Training step complete

2. For each server:
   a. Stop server, load new weights, restart
   b. POST /api/models [{model: "qwen-7b", endpoint: "<this>", version: 43}]
      → upsert: this server now at v43, others still at v42
      → gateway keeps routing to available servers

3. Eventually all servers at v43
```

## During the unavailable window

When no servers are registered for a model:

- Gateway returns **503 Service Unavailable** with `Retry-After` header
- Standard OpenAI SDKs (Python, JS) **auto-retry** on 503 with exponential backoff
- The agent pod **stays alive** — no crash, no K8s Job retry consumed
- When servers are re-registered, the next retry succeeds transparently

This is the key insight: the weight update window is invisible to agents. They experience a brief delay, not a failure.

## Per-request version tracking

Every `model_request` event records `server.version` — the training step of the model that generated the response:

```
[0]  model_request  {server.version: 42}   ← turn 1, policy v42
[1]  tool_result    {...}
[2]  model_request  {server.version: 42}   ← turn 2, policy v42
[3]  tool_result    {...}
     ── weight update: v42 → v43 ──
[4]  model_request  {server.version: 43}   ← turn 3, policy v43
[5]  reward         {value: 0.85}
```

This per-request tracking is essential for:

| Use case | How version helps |
|----------|-------------------|
| **Importance sampling** | Correct policy gradient when data comes from multiple policy versions |
| **Off-policy correction** | Adjust gradients for stale data |
| **Data filtering** | Discard or down-weight data from very old versions |
| **Metrics** | Track performance evolution across training steps |

## API summary

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/models` | Register servers. Body: `[{model, endpoint, version, token?}]`. Upsert by `(model, endpoint)`. |
| `GET` | `/api/models` | List all registered servers. |
| `DELETE` | `/api/models/{model}` | Remove servers for a model. Optional body: `{endpoints: [...]}` for specific servers. |
| `DELETE` | `/api/models` | Remove **all** servers. Gateway enters unavailable state. |
