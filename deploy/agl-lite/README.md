# agl-lite HTTP Service

Combines the LLM gateway proxy and the data store in a single HTTP service.

## Build

From repo root:
```bash
minikube image build -t agl-lite:dev -f deploy/agl-lite/Dockerfile .
```

## Deploy

```bash
source deploy/.env
kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/agl-lite/k8s.yaml
```

## Run locally (dev)

```bash
export AGL_KEY=dev-key
agl-lite serve --host 0.0.0.0 --port 8080
```

The same image is used for the controller (different command). See `deploy/controller/`.
