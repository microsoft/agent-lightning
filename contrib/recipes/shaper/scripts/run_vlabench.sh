#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$recipe_dir/../../.." && pwd)"

# Configuration. Edit these defaults or export the variables before running.
export SHAPER_BENCH_ROOT="${SHAPER_BENCH_ROOT:-$repo_root/shaper-benchmarks}"
export SHAPER_PLANNER_ENDPOINT="${SHAPER_PLANNER_ENDPOINT:-http://127.0.0.1:8001/v1}"
export SHAPER_MODEL="${SHAPER_MODEL:-Qwen/Qwen3.6-27B}"
export SHAPER_API_KEY_ENV="${SHAPER_API_KEY_ENV:-OPENAI_API_KEY}"
export SHAPER_PLANNER_MAX_TOKENS="${SHAPER_PLANNER_MAX_TOKENS:-32768}"
export SHAPER_OPTIMIZER_MAX_TOKENS="${SHAPER_OPTIMIZER_MAX_TOKENS:-65536}"
export SHAPER_PLANNER_TIMEOUT="${SHAPER_PLANNER_TIMEOUT:-300}"
export SHAPER_PLANNER_RETRIES="${SHAPER_PLANNER_RETRIES:-2}"
export SHAPER_PLANNER_TEMPERATURE="${SHAPER_PLANNER_TEMPERATURE:-1.0}"
export SHAPER_PLANNER_TOP_P="${SHAPER_PLANNER_TOP_P:-0.95}"
export SHAPER_PLANNER_PRESENCE_PENALTY="${SHAPER_PLANNER_PRESENCE_PENALTY:-0.0}"
export SHAPER_PLANNER_EXTRA_BODY="${SHAPER_PLANNER_EXTRA_BODY:-{\"top_k\":20,\"min_p\":0.0,\"repetition_penalty\":1.0}}"

export VLABENCH_ROOT="${VLABENCH_ROOT:-$SHAPER_BENCH_ROOT/VLABench/VLABench}"
export VLABENCH_TRACK="${VLABENCH_TRACK:-track_4_semantic_instruction}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export VLABENCH_VLA_HOST="${VLABENCH_VLA_HOST:-127.0.0.1}"
export VLABENCH_VLA_PORT="${VLABENCH_VLA_PORT:-8000}"
export VLABENCH_ACTOR_ID="${VLABENCH_ACTOR_ID:-vlabench-base}"
export VLABENCH_OPENPI_POLICY_CONFIG="${VLABENCH_OPENPI_POLICY_CONFIG:-pi0_ft_vlabench_primitive}"
export VLABENCH_OBSERVATION_SCHEMA="${VLABENCH_OBSERVATION_SCHEMA:-reported_three_camera}"
export VLABENCH_VLA_REPLAN_STEPS="${VLABENCH_VLA_REPLAN_STEPS:-5}"
export VLABENCH_VLA_TIMEOUT="${VLABENCH_VLA_TIMEOUT:-300}"
export VLABENCH_MAX_STEPS="${VLABENCH_MAX_STEPS:-400}"
export VLABENCH_MAX_VLM_ROUNDS="${VLABENCH_MAX_VLM_ROUNDS:-10}"
export VLABENCH_DEFAULT_ROUND_STEPS="${VLABENCH_DEFAULT_ROUND_STEPS:-200}"
export VLABENCH_MIN_ROUND_STEPS="${VLABENCH_MIN_ROUND_STEPS:-1}"
export VLABENCH_MAX_SUBSTEPS="${VLABENCH_MAX_SUBSTEPS:-1}"
export VLABENCH_RESET_WAIT_STEPS="${VLABENCH_RESET_WAIT_STEPS:-10}"

export SHAPER_HARNESS_TIMEOUT="${SHAPER_HARNESS_TIMEOUT:-3}"
export SHAPER_HARNESS_MEMORY_MB="${SHAPER_HARNESS_MEMORY_MB:-768}"
export SHAPER_HARNESS_MAX_OUTPUT_CHARS="${SHAPER_HARNESS_MAX_OUTPUT_CHARS:-32000000}"

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "Set PYTHON to a Python interpreter." >&2
  exit 2
fi
output_dir="${SHAPER_OUTPUT_DIR:-$repo_root/outputs/shaper/vlabench}"
n_runners="${SHAPER_N_RUNNERS:-1}"
cd "$repo_root"
command="${1:-}"
if [[ -n "$command" ]]; then
  shift
fi

usage() {
  cat <<EOF
Usage: $0 {check|train|eval} [additional arguments]

Defaults are in the Configuration block at the top of this file.
Set OPENAI_API_KEY (or the variable named by SHAPER_API_KEY_ENV) for a
remote authenticated planner endpoint.
EOF
}

case "$command" in
  check)
    exec "$python_bin" -m contrib.recipes.shaper.vlabench.check_env "$@"
    ;;
  train)
    exec "$python_bin" -m contrib.recipes.shaper.vlabench.train \
      --model "$SHAPER_MODEL" \
      --output-dir "$output_dir" \
      --n-runners "$n_runners" \
      "$@"
    ;;
  eval)
    skill_path="${SHAPER_SKILL_PATH:-$output_dir/best_skill.txt}"
    harness_path="${SHAPER_HARNESS_PATH:-$output_dir/best_harness.py}"
    exec "$python_bin" -m contrib.recipes.shaper.vlabench.evaluate \
      --split validation \
      --skill-path "$skill_path" \
      --harness-path "$harness_path" \
      --n-runners "$n_runners" \
      --output "$output_dir/evaluation.json" \
      "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
