# Getting Started with agl-lite

This guide walks through the setup and first run of agl-lite. It assumes you have a working K8s cluster and a compute backend (model server) already running.

> **Scope**: This is a setup guide, not an install guide. agl-lite is a Python HTTP service — install it however you like (pip, Docker, etc.). This doc covers the steps to wire everything together.

## Prerequisites

- A Kubernetes cluster (minikube for single-machine dev)
- `kubectl` configured to access the cluster
- A model server (vLLM, TGI, etc.) serving an OpenAI-compatible API
- An agent container image pushed to a registry accessible from the cluster

## Setup Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: K8s cluster ready                                  │
│  Step 2: Create namespace + RBAC + Secrets                  │
│  Step 3: Start agl-lite service                             │
│  Step 4: Start K8s controller                               │
│  Step 5: Register resources (job_defaults)                  │
│  Step 6: Register model servers                             │
│  Step 7: Enqueue rollouts                                   │
└─────────────────────────────────────────────────────────────┘
     Steps 1-4: one-time infra setup (DevOps)
     Steps 5-7: per-experiment (Algorithm / Researcher)
```

---

## Step 1: Ensure K8s Cluster

For local development, use minikube:

```bash
minikube start --cpus=4 --memory=8g
```

For production, any K8s cluster works (EKS, GKE, AKS, bare-metal).

Verify:
```bash
kubectl cluster-info
```

---

## Step 2: Create Namespace, RBAC, and Secrets

### 2a. Namespace (optional but recommended)

```bash
kubectl create namespace agl
```

### 2b. API Keys

Generate keys for the agent and controller roles. The algorithm key is also generated here — the algorithm always uses it when calling agl-lite, but may receive it through its own configuration rather than K8s Secret mount (e.g., if the algorithm runs outside the cluster).

```bash
kubectl -n agl create secret generic agl-lite-keys \
  --from-literal=AGENT_KEY=$(openssl rand -hex 32) \
  --from-literal=CONTROLLER_KEY=$(openssl rand -hex 32) \
  --from-literal=ALGORITHM_KEY=$(openssl rand -hex 32)
```

### 2c. Controller ServiceAccount + RBAC

The controller needs K8s API access to manage Jobs and read Secrets.

```yaml
# controller-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agl-controller
  namespace: agl
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: agl-controller-role
  namespace: agl
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]        # for find_succeeded_pod_uid
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get"]                # to read agl-lite-keys
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agl-controller-binding
  namespace: agl
subjects:
  - kind: ServiceAccount
    name: agl-controller
    namespace: agl
roleRef:
  kind: Role
  name: agl-controller-role
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f controller-rbac.yaml
```

---

## Step 3: Start agl-lite Service

agl-lite is a standalone HTTP server. It has **zero K8s dependency** — run it anywhere that's network-reachable from the cluster.

### Option A: Run locally (dev)

```bash
# Pass keys for verification (read from the Secret, or generate matching ones)
export AGL_AGENT_KEY="<same key as in K8s Secret>"
export AGL_CONTROLLER_KEY="<same key as in K8s Secret>"
export AGL_ALGORITHM_KEY="<same key as in K8s Secret>"

agl-lite serve --host 0.0.0.0 --port 8080
```

If running locally with minikube, the service needs to be reachable from inside the cluster. Use `minikube tunnel` or run agl-lite inside the cluster (Option B).

### Option B: Run in K8s (recommended)

```yaml
# agl-lite-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agl-lite
  namespace: agl
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agl-lite
  template:
    metadata:
      labels:
        app: agl-lite
    spec:
      containers:
        - name: agl-lite
          image: agl-lite:latest
          ports:
            - containerPort: 8080
          env:
            - name: AGL_AGENT_KEY
              valueFrom:
                secretKeyRef:
                  name: agl-lite-keys
                  key: AGENT_KEY
            - name: AGL_CONTROLLER_KEY
              valueFrom:
                secretKeyRef:
                  name: agl-lite-keys
                  key: CONTROLLER_KEY
            - name: AGL_ALGORITHM_KEY
              valueFrom:
                secretKeyRef:
                  name: agl-lite-keys
                  key: ALGORITHM_KEY
---
apiVersion: v1
kind: Service
metadata:
  name: agl-lite
  namespace: agl
spec:
  selector:
    app: agl-lite
  ports:
    - port: 8080
      targetPort: 8080
```

```bash
kubectl apply -f agl-lite-deployment.yaml
```

The internal URL is: `http://agl-lite.agl.svc.cluster.local:8080`

---

## Step 4: Start K8s Controller

The controller watches the agl-lite Store for `queuing` rollouts and creates K8s Jobs. It needs:
- **agl-lite URL** — to query rollouts and update status
- **K8s API access** — via the ServiceAccount from Step 2c
- **Controller API key** — from the Secret

### Option A: Run in K8s (recommended)

```yaml
# controller-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agl-controller
  namespace: agl
spec:
  replicas: 1                        # single instance — leader election not needed for MVP
  selector:
    matchLabels:
      app: agl-controller
  template:
    metadata:
      labels:
        app: agl-controller
    spec:
      serviceAccountName: agl-controller
      containers:
        - name: controller
          image: agl-lite-controller:latest
          env:
            - name: AGL_LITE_URL
              value: "http://agl-lite.agl.svc.cluster.local:8080"
            - name: AGL_CONTROLLER_KEY
              valueFrom:
                secretKeyRef:
                  name: agl-lite-keys
                  key: CONTROLLER_KEY
            - name: AGL_NAMESPACE
              value: "agl"
```

```bash
kubectl apply -f controller-deployment.yaml
```

### Option B: Run locally (dev)

```bash
export AGL_LITE_URL="http://localhost:8080"
export AGL_CONTROLLER_KEY="<controller key>"
export KUBECONFIG=~/.kube/config

agl-lite controller --namespace agl
```

---

**Infrastructure is ready.** Steps 1-4 are one-time. Everything below is per-experiment.

---

## Step 5: Register Resources (job_defaults)

The algorithm (or a setup script) posts a resource snapshot containing infra-level Job defaults. This tells the controller how to configure agent pods.

```python
import httpx

AGL_LITE_URL = "http://agl-lite.agl.svc.cluster.local:8080"
HEADERS = {"Authorization": "Bearer <ALGORITHM_KEY>"}

# Post resource snapshot with job_defaults
res = httpx.post(f"{AGL_LITE_URL}/api/resources", headers=HEADERS, json={
    "job_defaults": {
        "resources": {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"}
        },
        "node_selector": {},
        "tolerations": [],
        "service_account": "default",
        "image_pull_secrets": ["registry-creds"]
    },
    # Optional: include other resources (prompts, eval configs, etc.)
    "system_prompt": "You are a helpful coding assistant..."
})

resources_id = res.json()["resources_id"]
print(f"Resources snapshot: {resources_id}")
```

---

## Step 6: Register Model Servers

Tell agl-lite where the model servers are. The gateway routes agent LLM calls to these endpoints.

```python
# Register a model server
httpx.post(f"{AGL_LITE_URL}/api/models", headers=HEADERS, json={
    "name": "vllm-0",
    "endpoint": "http://vllm-server:8000",
    "model": "deepseek-r1-7b"
})

# Verify
models = httpx.get(f"{AGL_LITE_URL}/api/models", headers=HEADERS).json()
print(f"Registered models: {models}")
```

---

## Step 7: Enqueue Rollouts

Submit tasks. Each rollout becomes a K8s Job running your agent container.

```python
# Enqueue a batch of rollouts
resp = httpx.post(f"{AGL_LITE_URL}/api/rollouts", headers=HEADERS, json={
    "resources_id": resources_id,       # from Step 5
    "config": {
        "image": "my-agent:latest",
        "command": ["python", "solve.py"],
        "environment_variables": {"MODE": "train"},
        "timeout": 600,
        "max_retries": 2
    },
    "rollouts": [
        {"input": {"prompt": "Write a function to sort a list"}},
        {"input": {"prompt": "Write a function to find duplicates"}},
        {"input": {"prompt": "Write a binary search implementation"}},
    ]
})

rollout_ids = [r["rollout_id"] for r in resp.json()]
print(f"Enqueued {len(rollout_ids)} rollouts")
```

---

## What Happens Next

1. The **controller** picks up `queuing` rollouts, fetches `job_defaults` from the resources snapshot, merges with `rollout.config`, and creates K8s Jobs.

2. Each **agent pod** starts with 4 env vars:
   - `OPENAI_BASE_URL` — points to agl-lite gateway (with rollout/attempt path prefix)
   - `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — agent auth key (from K8s Secret, same key for both)
   - `AGL_TASK_INPUT` — JSON task payload
   - `AGL_EVENT_URL` — for posting custom events (optional)

3. The agent runs, makes LLM calls through `OPENAI_BASE_URL`. The gateway:
   - Validates the rollout exists
   - Forwards to a registered model server
   - Captures request + response as `model_request` events

4. When the agent finishes (or fails/times out), the controller updates rollout status.

5. The **algorithm** polls for completion and retrieves trajectories:

```python
import time

# Poll for completion
while True:
    rollouts = httpx.get(f"{AGL_LITE_URL}/api/rollouts",
        params={"ids": ",".join(rollout_ids)}, headers=HEADERS).json()
    
    done = all(r["status"] in ("succeeded", "terminal_failed", "cancelled")
               for r in rollouts)
    if done:
        break
    time.sleep(5)

# Retrieve events (trajectories) for succeeded rollouts
for r in rollouts:
    if r["status"] == "succeeded":
        events = httpx.get(f"{AGL_LITE_URL}/api/events",
            params={"rollout_id": r["rollout_id"]}, headers=HEADERS).json()
        print(f"Rollout {r['rollout_id']}: {len(events)} events")
```

---

## Summary

| Step | Who | What | One-time? |
|------|-----|------|-----------|
| 1. K8s cluster | DevOps | Ensure cluster exists | ✅ |
| 2. Namespace + RBAC + Secrets | DevOps | Create namespace, ServiceAccount, API keys | ✅ |
| 3. Start agl-lite | DevOps | Run the HTTP service | ✅ |
| 4. Start controller | DevOps | Run the K8s controller | ✅ |
| 5. Register resources | Researcher | Post job_defaults + prompts | Per experiment |
| 6. Register model servers | Researcher | Tell gateway where models are | Per experiment |
| 7. Enqueue rollouts | Researcher | Submit tasks | Per batch |
