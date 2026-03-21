# Deploying agl-lite

## Structure

```
deploy/
├── agl-lite/              # HTTP service (store + gateway)
│   ├── Dockerfile         # builds agl-lite:dev image
│   ├── k8s.yaml           # Deployment + Service
│   └── README.md
├── controller/            # K8s reconciler (reuses agl-lite image)
│   ├── k8s.yaml           # Deployment
│   ├── rbac.yaml          # ServiceAccount + Role + RoleBinding
│   └── README.md
├── config.example.yaml    # all non-secret config (copy to config.yaml)
└── README.md              # this file
```

## Configuration

- **`deploy/config.yaml`** — all non-secret config (namespace, URLs, controller settings). Structured YAML, version-controllable.
- **`AGL_KEY` env var** — the only secret. Never on disk. Set before deploying.

## Quick Start

```bash
# 1. Copy and edit config
cp deploy/config.example.yaml deploy/config.yaml

# 2. Build image
scripts/build_images.sh

# 3. Set secret and deploy
export AGL_KEY=$(openssl rand -hex 32)
python scripts/deploy.py

# 4. Access from host
kubectl -n agl port-forward svc/agl-lite 8080:8080
export AGL_LITE_URL=http://localhost:8080
agl-client health
```

## Teardown

```bash
python scripts/deploy.py --teardown
```
