# Deploying agl-lite

## Structure

```
deploy/
├── agl-lite/              # HTTP service (store + gateway)
│   ├── Dockerfile
│   ├── k8s.yaml           # Deployment + Service
│   └── README.md
├── controller/            # K8s reconciler (reuses agl-lite image)
│   ├── job-template.yaml.j2   # Default Jinja2 job manifest template
│   ├── k8s.yaml               # Deployment
│   ├── rbac.yaml              # ServiceAccount + Role + RoleBinding
│   └── README.md
├── agl-lite.env.example  # Deploy config template (copy and edit)
└── README.md              # this file
```

## Configuration

All deploy config lives in a `.env` file. Copy the example and edit:

```bash
cp deploy/agl-lite.env.example deploy/agl-lite.env
$EDITOR deploy/agl-lite.env
```

This file is also the **single project config** — add hook config, model
endpoints, and experiment parameters here. Extra variables are silently ignored
by the deploy command and consumed by other components via `os.environ`.

Set the API key via environment variable (never in the config file):

```bash
export AGL_KEY=$(openssl rand -hex 32)
```

## Quick Start

```bash
# 1. Copy and edit config
cp deploy/agl-lite.env.example deploy/agl-lite.env

# 2. Build image
scripts/build_images.sh

# 3. Deploy
export AGL_KEY=$(openssl rand -hex 32)
agl-lite deploy --env-file deploy/agl-lite.env

# 4. Source the generated env file for host-side access
source .local/agl-lite.env
agl-client health
```

## Cleanup

```bash
agl-lite deploy --env-file deploy/agl-lite.env --cleanup
```

See [docs/deploy.md](../docs/deploy.md) for full configuration reference.
