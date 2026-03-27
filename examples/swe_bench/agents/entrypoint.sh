#!/bin/bash
# Shared entrypoint for SWE-bench agent containers.
#
# Runs inside the official SWE-bench Docker image at /testbed.
# Dispatches to the agent specified by AGL_CODING_AGENT env var,
# then runs eval_script, and hands off to grade.py for reporting.
#
# Output flow:
#   1. All outputs written to OUTPUT_DIR (patch.diff, test_output.txt)
#   2. grade.py posts agent_output event, grades, posts reward event
#   3. Shell copies OUTPUT_DIR → ARTIFACT_DIR for archival
#
# Expected env vars (set by hook via on_enqueue):
#   AGL_TASK_INPUT      — problem statement
#   AGL_EVAL_SCRIPT     — bash script to run tests
#   AGL_EVAL_META       — JSON with FAIL_TO_PASS, PASS_TO_PASS, instance_id, repo, version
#   AGL_CODING_AGENT    — agent name (e.g., "claude_code")
#
# Expected env vars (set by controller):
#   AGL_EVENT_URL       — URL to post events
#   AGL_ROLLOUT_ID      — rollout ID
#   AGL_POD_UID         — pod UID (used as attempt ID)
#   OPENAI_BASE_URL     — LLM proxy URL
#   OPENAI_API_KEY      — API key

set -euo pipefail

AGENT_DIR="/agl/agents"
AGENT_NAME="${AGL_CODING_AGENT:-claude_code}"

# Local output directory — all files go here first.
OUTPUT_DIR="/tmp/agl_output"
mkdir -p "$OUTPUT_DIR"

# Final artifact path: ARTIFACT_ROOT / rollout_id / attempt_id
ARTIFACT_ROOT="/data/artifacts"
ROLLOUT_ID="${AGL_ROLLOUT_ID:-unknown}"
ATTEMPT_ID="${AGL_POD_UID:-unknown}"
ARTIFACT_DIR="${ARTIFACT_ROOT}/${ROLLOUT_ID}/${ATTEMPT_ID}"

echo "=== SWE-bench Agent Entrypoint ==="
echo "  Agent: $AGENT_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Working dir: /testbed"

cd /testbed

# ── Phase 1: Install + run the coding agent ──────────────────────
echo "--- Phase 1: Agent ---"

if [ -f "$AGENT_DIR/${AGENT_NAME}--install.sh" ]; then
    echo "Installing agent: $AGENT_NAME"
    bash "$AGENT_DIR/${AGENT_NAME}--install.sh"
fi

if [ -f "$AGENT_DIR/${AGENT_NAME}--run.sh" ]; then
    echo "Running agent: $AGENT_NAME"
    bash "$AGENT_DIR/${AGENT_NAME}--run.sh" || echo "WARNING: Agent exited with code $?"
else
    echo "ERROR: No run script found for agent $AGENT_NAME"
    echo "  Expected: $AGENT_DIR/${AGENT_NAME}--run.sh"
    exit 1
fi

# ── Phase 2: Capture patch ───────────────────────────────────────
echo "--- Phase 2: Capture patch ---"

git -c core.fileMode=false diff HEAD > "$OUTPUT_DIR/patch.diff" 2>/dev/null || true
echo "Patch size: $(wc -c < "$OUTPUT_DIR/patch.diff") bytes"

# ── Phase 3: Run eval_script ─────────────────────────────────────
echo "--- Phase 3: Evaluate ---"

if [ -n "${AGL_EVAL_SCRIPT:-}" ]; then
    echo "$AGL_EVAL_SCRIPT" > /tmp/eval.sh
    chmod +x /tmp/eval.sh
    bash /tmp/eval.sh > "$OUTPUT_DIR/test_output.txt" 2>&1 || true
    echo "Test output: $(wc -c < "$OUTPUT_DIR/test_output.txt") bytes"
else
    echo "WARNING: No AGL_EVAL_SCRIPT set, skipping evaluation"
    echo "No eval script provided" > "$OUTPUT_DIR/test_output.txt"
fi

# ── Phase 4: Report + Grade ──────────────────────────────────────
echo "--- Phase 4: Report + Grade ---"

python3 -m pip install swebench -q 2>/dev/null || python3 -m pip install swebench 2>&1 | tail -1

python3 /agl/agents/grade.py "$OUTPUT_DIR" "${ROLLOUT_ID}/${ATTEMPT_ID}"

# ── Phase 5: Archive outputs ─────────────────────────────────────
echo "--- Phase 5: Archive ---"

mkdir -p "$ARTIFACT_DIR"
cp -r "$OUTPUT_DIR"/. "$ARTIFACT_DIR"/ 2>/dev/null \
    && echo "Archived outputs to $ARTIFACT_DIR" \
    || echo "WARNING: Failed to archive outputs (volume not mounted?)"

echo "=== Entrypoint complete ==="
