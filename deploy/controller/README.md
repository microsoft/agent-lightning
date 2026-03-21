# K8s Controller

Reconciles rollouts from the agl-lite store into K8s Jobs. Reuses the `agl-lite:dev` image with a different command.

## Deploy

```bash
source deploy/.env
kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/controller/rbac.yaml
kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/controller/k8s.yaml
```

## Run locally (dev)

```bash
export AGL_KEY=dev-key
export AGL_LITE_URL=http://localhost:8080
agl-lite controller --namespace agl --secret-name agl-lite-keys
```

No separate Docker image needed — same image as agl-lite serve.
