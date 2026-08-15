#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail

VLABENCH_COMMIT="cf588fe60c0c7282174fe979f5913170cfe69017"
OPENPI_COMMIT="4483d1da6332da44115fe530e4e6fdd89bd57b13"
ESI_BENCH_COMMIT="3c1756396f32b1a90c1f72356a7fde45f418e179"
BEHAVIOR_COMMIT="67ad490856dd465d4606663106f81673fc8bf4e8"
GOOGLE_GENAI_VERSION="1.75.0"
FASTAPI_VERSION="0.121.2"
STARLETTE_VERSION="0.49.3"

usage() {
  cat <<'EOF'
Usage:
  bootstrap_shaper_environment.sh common AGL_ROOT
  bootstrap_shaper_environment.sh vlabench-simulator AGL_ROOT VLABENCH_CHECKOUT OPENPI_ROOT [--download-assets]
  bootstrap_shaper_environment.sh vlabench-actor AGL_ROOT OPENPI_ROOT CHECKPOINT_DIR [--download-checkpoint]
  bootstrap_shaper_environment.sh esi-controller AGL_ROOT
  bootstrap_shaper_environment.sh esi-worker AGL_ROOT ESI_ROOT BEHAVIOR_ROOT [--install-behavior]

Run each mode inside its intended Python environment. The ESI worker mode
requires an active Python 3.11 conda environment plus explicit acceptance:
  SHAPER_ACCEPT_NVIDIA_EULA=YES
  SHAPER_ACCEPT_BEHAVIOR_DATASET_TOS=YES
and requires OMNIGIBSON_DATA_PATH to point at a data disk.
EOF
}

die() {
  echo "[shaper-bootstrap] $*" >&2
  exit 2
}

python_bin() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
  elif command -v python >/dev/null 2>&1; then
    command -v python
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    die "No Python interpreter found. Set PYTHON to the target environment interpreter."
  fi
}

absolute_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing directory: $path"
  (cd "$path" && pwd)
}

require_commit() {
  local checkout="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD 2>/dev/null)" || die "$label is not a Git checkout: $checkout"
  [[ "$actual" == "$expected" ]] || die "$label revision $actual does not match pinned $expected"
}

require_python_version() {
  local executable="$1"
  local expected="$2"
  local actual
  actual="$($executable -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$actual" == "$expected" ]] || die "Expected Python $expected, found $actual at $executable"
}

install_agl() {
  local executable="$1"
  local agl_root="$2"
  "$executable" -m pip install -e "$agl_root"
  "$executable" -m pip install -e "$agl_root/contrib/agentlightning/contrib/shaper"
  # Editable installs resolve the root dependency set and can otherwise
  # upgrade these packages again. Pin them last to the versions validated by
  # the repository lock; newer FastAPI releases remove APIs used by LiteLLM.
  "$executable" -m pip install --upgrade --force-reinstall \
    "fastapi==$FASTAPI_VERSION" "starlette==$STARLETTE_VERSION"
  "$executable" -c 'from agentlightning.contrib.shaper import SHAPER; print("SHAPER import:", SHAPER.__name__)'
}

mode="${1:-}"
if [[ -z "$mode" ]]; then
  usage
  exit 2
fi
shift

case "$mode" in
  common)
    [[ "$#" -eq 1 ]] || { usage; exit 2; }
    agl_root="$(absolute_dir "$1")"
    py="$(python_bin)"
    install_agl "$py" "$agl_root"
    ;;

  vlabench-simulator)
    [[ "$#" -ge 3 && "$#" -le 4 ]] || { usage; exit 2; }
    agl_root="$(absolute_dir "$1")"
    vlabench_checkout="$(absolute_dir "$2")"
    openpi_root="$(absolute_dir "$3")"
    option="${4:-}"
    [[ -z "$option" || "$option" == "--download-assets" ]] || die "Unknown option: $option"
    require_commit "$vlabench_checkout" "$VLABENCH_COMMIT" "VLABench"
    require_commit "$openpi_root" "$OPENPI_COMMIT" "OpenPI"
    py="$(python_bin)"
    require_python_version "$py" "3.10"
    export VLABENCH_ROOT="$vlabench_checkout/VLABench"
    "$py" -m pip install -r "$agl_root/contrib/recipes/shaper/vlabench/requirements-simulator.txt"
    "$py" -m pip install --no-deps -e "$vlabench_checkout"
    "$py" -m pip install -e "$openpi_root/packages/openpi-client"
    install_agl "$py" "$agl_root"
    if [[ "$option" == "--download-assets" ]]; then
      (cd "$vlabench_checkout" && "$py" scripts/download_assets.py --choice all)
    fi
    PYTHONPATH="$agl_root${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$VLABENCH_ROOT" <<'PY'
from pathlib import Path
import sys

from contrib.recipes.shaper.vlabench.check_env import check_vlabench_assets

errors = check_vlabench_assets(Path(sys.argv[1]))
if errors:
    raise SystemExit("\n".join(errors) + "\nRe-run with --download-assets.")
import VLABench  # noqa: F401
import openpi_client  # noqa: F401
print("VLABench simulator imports and assets passed.")
PY
    ;;

  vlabench-actor)
    [[ "$#" -ge 3 && "$#" -le 4 ]] || { usage; exit 2; }
    agl_root="$(absolute_dir "$1")"
    openpi_root="$(absolute_dir "$2")"
    checkpoint_dir="$3"
    option="${4:-}"
    [[ -z "$option" || "$option" == "--download-checkpoint" ]] || die "Unknown option: $option"
    require_commit "$openpi_root" "$OPENPI_COMMIT" "OpenPI"
    command -v uv >/dev/null 2>&1 || die "uv is required for the pinned OpenPI environment."
    (cd "$openpi_root" && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev)
    actor_python="$openpi_root/.venv/bin/python"
    [[ -x "$actor_python" ]] || die "OpenPI uv environment did not create $actor_python"
    if [[ "$option" == "--download-checkpoint" ]]; then
      PYTHONPATH="$agl_root${PYTHONPATH:+:$PYTHONPATH}" \
        "$actor_python" "$agl_root/contrib/recipes/shaper/scripts/download_shaper_vlabench_actor.py" \
        "$checkpoint_dir"
    else
      PYTHONPATH="$agl_root${PYTHONPATH:+:$PYTHONPATH}" \
        "$actor_python" "$agl_root/contrib/recipes/shaper/scripts/download_shaper_vlabench_actor.py" \
        "$checkpoint_dir" --verify-only
    fi
    PYTHONPATH="$agl_root${PYTHONPATH:+:$PYTHONPATH}" "$actor_python" -c \
      'import openpi; import contrib.recipes.shaper.vlabench.openpi_server; print("OpenPI actor imports passed.")'
    ;;

  esi-controller)
    [[ "$#" -eq 1 ]] || { usage; exit 2; }
    agl_root="$(absolute_dir "$1")"
    py="$(python_bin)"
    install_agl "$py" "$agl_root"
    ;;

  esi-worker)
    [[ "$#" -ge 3 && "$#" -le 4 ]] || { usage; exit 2; }
    agl_root="$(absolute_dir "$1")"
    esi_root="$(absolute_dir "$2")"
    behavior_root="$(absolute_dir "$3")"
    option="${4:-}"
    [[ -z "$option" || "$option" == "--install-behavior" ]] || die "Unknown option: $option"
    [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || die "ESI-Bench requires Linux x86_64."
    require_commit "$esi_root" "$ESI_BENCH_COMMIT" "ESI-Bench"
    require_commit "$behavior_root" "$BEHAVIOR_COMMIT" "BEHAVIOR-1K"
    py="$(python_bin)"
    require_python_version "$py" "3.11"
    [[ -n "${CONDA_PREFIX:-}" ]] || die "Activate the target behavior conda environment first."
    case "$py" in
      "$CONDA_PREFIX"/*) ;;
      *) die "PYTHON resolves outside active CONDA_PREFIX=$CONDA_PREFIX: $py" ;;
    esac
    map_patch="$agl_root/contrib/recipes/shaper/esi_bench/patches/behavior_floor_maps.patch"
    if git -C "$behavior_root" apply --check "$map_patch" >/dev/null 2>&1; then
      git -C "$behavior_root" apply "$map_patch"
    elif ! git -C "$behavior_root" apply --reverse --check "$map_patch" >/dev/null 2>&1; then
      die "The official ESI map patch cannot be applied cleanly to $behavior_root"
    fi
    if [[ "$option" == "--install-behavior" ]]; then
      [[ "${SHAPER_ACCEPT_NVIDIA_EULA:-}" == "YES" ]] || \
        die "Set SHAPER_ACCEPT_NVIDIA_EULA=YES after reviewing the NVIDIA Isaac Sim EULA."
      [[ "${SHAPER_ACCEPT_BEHAVIOR_DATASET_TOS:-}" == "YES" ]] || \
        die "Set SHAPER_ACCEPT_BEHAVIOR_DATASET_TOS=YES after reviewing the BEHAVIOR dataset terms."
      [[ -n "${OMNIGIBSON_DATA_PATH:-}" ]] || die "Set OMNIGIBSON_DATA_PATH to a large data disk."
      mkdir -p "$OMNIGIBSON_DATA_PATH"
      if [[ ! -x "$CONDA_PREFIX/bin/pip" ]]; then
        "$py" -m ensurepip --upgrade
      fi
      (
        cd "$behavior_root"
        export PATH="$CONDA_PREFIX/bin:$PATH"
        [[ "$(command -v python)" == "$CONDA_PREFIX/bin/python" ]] || \
          die "BEHAVIOR setup would use Python outside the active conda environment."
        [[ "$(command -v pip)" == "$CONDA_PREFIX/bin/pip" ]] || \
          die "BEHAVIOR setup would use pip outside the active conda environment."
        bash setup.sh \
        --omnigibson --bddl --dataset \
        --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos \
        --confirm-no-conda --cuda-version "${SHAPER_BEHAVIOR_CUDA_VERSION:-12.8}"
      )
    fi
    "$py" -m pip install "google-genai==$GOOGLE_GENAI_VERSION"
    PYTHONPATH="$agl_root${PYTHONPATH:+:$PYTHONPATH}" \
      ESI_BEHAVIOR_ROOT="$behavior_root" ESI_BENCH_ROOT="$esi_root" \
      ESI_OMNIGIBSON_DATA_ROOT="${OMNIGIBSON_DATA_PATH:-${ESI_OMNIGIBSON_DATA_ROOT:-}}" \
      "$py" - <<'PY'
from pathlib import Path
import os

from contrib.recipes.shaper.esi_bench.check_env import (
    check_map_generation_patch,
    check_omnigibson_assets,
    check_runtime_modules,
)
from contrib.recipes.shaper.esi_bench.contracts import check_behavior_source, check_omnigibson_install

behavior = Path(os.environ["ESI_BEHAVIOR_ROOT"])
data = Path(os.environ.get("ESI_OMNIGIBSON_DATA_ROOT", ""))
errors = [
    *check_behavior_source(behavior),
    *check_omnigibson_install(behavior),
    *check_runtime_modules(),
    *check_map_generation_patch(
        behavior / "asset_pipeline" / "b1k_pipeline" / "usd_conversion" / "make_maps.py"
    ),
    *check_omnigibson_assets(data),
]
if errors:
    raise SystemExit("\n".join(errors))
print("ESI-Bench behavior environment, modules, map setting, and assets passed.")
PY
    ;;

  *)
    usage
    exit 2
    ;;
esac
