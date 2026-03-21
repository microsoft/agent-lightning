# Deploying agl-lite

## Structure

```
deploy/
├── agl-lite/          # HTTP service (store + gateway)
│   ├── Dockerfile     # builds agl-lite:dev image
│   ├── k8s.yaml       # Deployment + Service
│   └── README.md
├── controller/        # K8s reconciler (reuses agl-lite image)
│   ├── k8s.yaml       # Deployment
│   ├── rbac.yaml      # ServiceAccount + Role + RoleBinding
│   └── README.md
├── .env.example       # secrets + bootstrap (AGL_KEY, AGL_K8S_NAMESPACE)
├── config.example.yaml # non-secret config (AGL_LITE_URL, controller settings)
└── README.md          # this file
```

## Quick Start

1. Copy and edit config files:
   ```bash
   cp deploy/.env.example deploy/.env
   cp deploy/config.example.yaml deploy/config.yaml
   # Edit both files with your values
   ```

2. Build image:
   ```bash
   scripts/build_images.sh
   ```

3. Deploy:
   ```bash
   scripts/deploy.sh
   ```

4. Teardown:
   ```bash
   scripts/deploy.sh --teardown
   ```

## Manual Deploy

```bash
source deploy/.env

# Create namespace
kubectl create namespace $AGL_K8S_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Create secret (AGL_KEY never on disk)
kubectl -n $AGL_K8S_NAMESPACE create secret generic agl-lite-keys \
  --from-literal=AGL_KEY="$AGL_KEY" --dry-run=client -o yaml | kubectl apply -f -

# Create configmap (YAML file + extracted values as env-accessible keys)
AGL_LITE_URL=$(grep '^agl_lite_url:' deploy/config.yaml | awk '{print $2}')
kubectl -n $AGL_K8S_NAMESPACE create configmap agl-lite-config \
  --from-file=config.yaml=deploy/config.yaml \
  --from-literal=AGL_K8S_NAMESPACE="$AGL_K8S_NAMESPACE" \
  --from-literal=AGL_LITE_URL="$AGL_LITE_URL" \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply manifests
kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/controller/rbac.yaml
kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/agl-lite/k8s.yaml
kubectl apply -n $AGL_K8S_NAMESPACE -f deploy/controller/k8s.yaml

# Wait for ready
kubectl -n $AGL_K8S_NAMESPACE wait --for=condition=available deployment/agl-lite --timeout=120s
kubectl -n $AGL_K8S_NAMESPACE wait --for=condition=available deployment/agl-controller --timeout=120s
```
