# Deploying agl-lite

## Structure

```
deploy/
├── agl-lite/              # HTTP service (store + gateway)
│   ├── Dockerfile
│   ├── k8s.yaml           # Deployment + Service
│   └── README.md
├── controller/            # K8s reconciler (reuses agl-lite image)
│   ├── k8s.yaml           # Deployment
│   ├── rbac.yaml          # ServiceAccount + Role + RoleBinding
│   └── README.md
├── .env.example           # all config (copy to .env)
└── README.md              # this file
```

## Configuration

- **`deploy/.env`** — all config: namespace, URLs, controller settings. AGL_KEY placeholder (commented out — set via env var or uncomment).
- **`AGL_KEY`** — either uncomment in `.env` or `export AGL_KEY=...` in your shell.

## Quick Start

```bash
# 1. Copy and edit config
cp deploy/.env.example deploy/.env

# 2. Build image
scripts/build_images.sh

# 3. Set secret and deploy
export AGL_KEY=$(openssl rand -hex 32)
scripts/deploy.sh

# 4. Access from host
kubectl -n agl port-forward svc/agl-lite 8080:8080
export AGL_BASE_URL=http://localhost:8080
agl-client health
```

## Cleanup

```bash
scripts/deploy.sh --cleanup
```
