#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 5 ]]; then
  echo "Usage: start_shaper_vlabench_actor.sh AGL_ROOT OPENPI_ROOT CHECKPOINT_DIR [PORT] [ACTOR_ID]" >&2
  exit 2
fi

agl_root="$(cd "$1" && pwd)"
openpi_root="$(cd "$2" && pwd)"
checkpoint_dir="$3"
port="${4:-8000}"
actor_id="${5:-vlabench-base}"
actor_python="$openpi_root/.venv/bin/python"

[[ -x "$actor_python" ]] || {
  echo "Missing OpenPI environment: $actor_python" >&2
  exit 2
}

# The actor may share one GPU with a small OpenAI-compatible planner. Disable
# JAX's default up-front reservation; this changes allocation, not inference.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYTHONPATH="$agl_root${PYTHONPATH:+:$PYTHONPATH}" exec "$actor_python" -m \
  contrib.recipes.shaper.vlabench.openpi_server \
  --openpi-root "$openpi_root" \
  --policy-config pi0_ft_vlabench_primitive \
  --policy-dir "$checkpoint_dir" \
  --actor-id "$actor_id" \
  --observation-schema reported_three_camera \
  --port "$port"
