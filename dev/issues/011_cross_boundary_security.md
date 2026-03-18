# 011 — 🟡 Cross-boundary Authentication and Transport Security

## Problem

The architecture states "no strong co-location assumptions" — agl-lite Service can run in a different cluster or cloud from the K8s controller and agent pods. When they're separated by network boundaries, three communication channels need securing:

1. **K8s controller → agl-lite Service** (rollout status updates, queries)
2. **Agent pods → agl-lite Service** (LLM proxy, event ingestion)
3. **Algorithm → agl-lite Service** (enqueue rollouts, query events, manage resources)

Without authentication, anyone with network access to the agl-lite endpoint can:
- Enqueue arbitrary rollouts
- Read all trajectory data (potentially sensitive — model outputs, user prompts)
- Modify rollout status (fake completions, cancel others' work)
- Inject fake events (corrupt training data)

## Threat model

| Scenario | Risk |
|----------|------|
| agl-lite exposed via public ingress | Full unauthorized access |
| agl-lite in VPC but cross-account | Lateral movement from compromised workload |
| Same cluster, different namespaces | Namespace escape or misconfigured NetworkPolicy |
| Same cluster, same namespace | Low risk — trust boundary is the cluster |

## What needs securing

### Transport: TLS

All cross-boundary traffic must use TLS. Within the same cluster, optional (K8s internal DNS is trusted).

### Identity: who is calling?

| Caller | Identity mechanism |
|--------|-------------------|
| K8s controller | Service account token (projected volume) or API key |
| Agent pods | Short-lived token scoped to `(rollout_id, attempt_id)` — injected as env var by controller |
| Algorithm | API key or mutual TLS (long-lived, privileged) |

### Authorization: what can they do?

| Role | Allowed operations |
|------|-------------------|
| **agent** | Only its own proxy path (`/rollout/{rid}/attempt/{aid}/...`) and event ingestion. Cannot query other rollouts, cannot update status. |
| **controller** | `PATCH /api/rollouts/{rid}`, `GET /api/rollouts`. Cannot read events or manage resources. |
| **algorithm** | Full access: enqueue, query events, manage resources, cancel. |

## Options

### A. API key (simple)

- Single shared secret per role (agent, controller, algorithm)
- Passed as `Authorization: Bearer <key>` header
- agl-lite validates key and maps to role
- Agent pods get key injected as env var from K8s Secret

**Pros**: simple, works everywhere, no infrastructure dependency.
**Cons**: shared secret rotation is painful; agent key is broad (any agent can hit any rollout's path).

### B. Scoped tokens (per-rollout)

- Controller generates a short-lived JWT when creating the Job, scoped to `(rollout_id, attempt_id)`
- Token injected as env var into agent pod
- agl-lite validates JWT signature and checks scope on every request
- Controller and Algorithm use long-lived API keys

**Pros**: least-privilege for agents; compromised pod can't access other rollouts.
**Cons**: agl-lite needs JWT verification; token generation adds complexity to controller.

### C. Mutual TLS (mTLS)

- Each component has a client certificate signed by a shared CA
- agl-lite verifies client cert and extracts role from CN/SAN
- No tokens or keys to manage

**Pros**: strong identity, no secret rotation (cert rotation via cert-manager).
**Cons**: certificate infrastructure overhead; harder to scope per-rollout.

### D. Network-level only (VPN / private link)

- No application-level auth
- Rely on network isolation (VPC peering, WireGuard, Tailscale)
- All callers within the network are trusted

**Pros**: zero application changes.
**Cons**: no defense in depth; any compromised workload in the network has full access.

## Recommendation

**A (API key) for MVP, with B (scoped tokens) as the production target.**

MVP:
- Three API keys: `AGL_AGENT_KEY`, `AGL_CONTROLLER_KEY`, `AGL_ALGORITHM_KEY`
- Stored in K8s Secrets, injected as env vars
- agl-lite checks `Authorization` header, maps key → role, enforces role-based access
- TLS via ingress controller or load balancer (not application-level)

Production path:
- Controller mints per-rollout JWT for agent pods (short TTL, scoped to `rid/aid`)
- Algorithm and controller keep API keys or move to mTLS
- Add rate limiting per role

## Changes needed

- Add Section 3.7 "Security" to architecture doc (or note in Section 3.4)
- Add `Authorization` header to API spec
- Add `AGL_AGENT_KEY` to Job template env vars
- Add role-based access table to API path layout
- Note TLS requirement for cross-boundary deployments

## Affected sections

- Section 3.2 (Component Mapping — mention auth requirement)
- Section 3.4 (Unified API Spec — add auth headers and role column)
- Section 3.5 (K8s Controller — Job template env vars)
