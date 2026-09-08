#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail

VLABENCH_REPOSITORY="${VLABENCH_REPOSITORY:-https://github.com/OpenMOSS/VLABench.git}"
VLABENCH_COMMIT="cf588fe60c0c7282174fe979f5913170cfe69017"
OPENPI_REPOSITORY="${OPENPI_REPOSITORY:-https://github.com/Shiduo-zh/openpi.git}"
OPENPI_COMMIT="4483d1da6332da44115fe530e4e6fdd89bd57b13"
ESI_BENCH_REPOSITORY="${ESI_BENCH_REPOSITORY:-https://github.com/ESI-Bench/ESI-Bench.git}"
ESI_BENCH_COMMIT="3c1756396f32b1a90c1f72356a7fde45f418e179"
BEHAVIOR_REPOSITORY="${BEHAVIOR_REPOSITORY:-https://github.com/StanfordVL/BEHAVIOR-1K.git}"
BEHAVIOR_COMMIT="67ad490856dd465d4606663106f81673fc8bf4e8"

destination="${1:-$PWD/shaper-benchmarks}"
mkdir -p "$destination"
destination="$(cd "$destination" && pwd)"

checkout_repository() {
  local repository="$1"
  local commit="$2"
  local target="$3"
  local created=0

  if [[ ! -d "$target/.git" ]]; then
    if [[ -e "$target" ]]; then
      if [[ ! -d "$target" || -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite non-empty path: $target" >&2
        return 2
      fi
    fi
    git clone --filter=blob:none --no-checkout "$repository" "$target"
    created=1
  fi
  # A fresh --no-checkout clone reports every tracked file as deleted until its
  # first checkout. Only protect pre-existing worktrees from mutation.
  if [[ "$created" -eq 0 && -n "$(git -C "$target" status --porcelain --untracked-files=all)" ]]; then
    echo "Refusing to change dirty benchmark checkout: $target" >&2
    return 2
  fi
  git -C "$target" fetch --depth=1 origin "$commit"
  git -C "$target" checkout --detach "$commit"
  test "$(git -C "$target" rev-parse HEAD)" = "$commit"
}

vlabench_target="$destination/VLABench"
openpi_target="$destination/OpenPI"
esi_target="$destination/ESI-Bench"
behavior_target="$destination/BEHAVIOR-1K"

checkout_repository "$VLABENCH_REPOSITORY" "$VLABENCH_COMMIT" "$vlabench_target"
checkout_repository "$OPENPI_REPOSITORY" "$OPENPI_COMMIT" "$openpi_target"

checkout_repository "$ESI_BENCH_REPOSITORY" "$ESI_BENCH_COMMIT" "$esi_target"
checkout_repository "$BEHAVIOR_REPOSITORY" "$BEHAVIOR_COMMIT" "$behavior_target"

printf 'VLABENCH_CHECKOUT=%s\n' "$vlabench_target"
printf 'VLABENCH_ROOT=%s\n' "$vlabench_target/VLABench"
printf 'OPENPI_ROOT=%s\n' "$openpi_target"
printf 'ESI_BENCH_ROOT=%s\n' "$esi_target"
printf 'ESI_BEHAVIOR_ROOT=%s\n' "$behavior_target"
printf '%s\n' "Source checkouts are pinned. Simulator assets, checkpoints, and Python environments are not downloaded."
