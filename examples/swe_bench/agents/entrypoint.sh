#!/bin/bash
# Shared entrypoint for SWE-bench agent containers.
#
# Runs inside the official SWE-bench Docker image at /testbed.
# Dispatches to the agent specified by AGL_CODING_AGENT env var,
# then runs eval_script, grades, posts reward, and archives outputs.
#
# Output flow:
#   1. All outputs written to local OUTPUT_DIR first
#   2. grade.py reads from OUTPUT_DIR, posts reward event
#   3. Shell copies OUTPUT_DIR → ARTIFACT_DIR for archival
#   4. agent_output event reports artifact_path (relative to ARTIFACT_ROOT)
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

# Agent scripts are mounted flat: <agent_name>--<script>.sh
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

PATCH_FILE="$OUTPUT_DIR/patch.diff"
git -c core.fileMode=false diff HEAD > "$PATCH_FILE" 2>/dev/null || echo -n "" > "$PATCH_FILE"
PATCH_SIZE=$(wc -c < "$PATCH_FILE")
echo "Patch size: $PATCH_SIZE bytes"

# ── Phase 3: Run eval_script ─────────────────────────────────────
echo "--- Phase 3: Evaluate ---"

TEST_OUTPUT_FILE="$OUTPUT_DIR/test_output.txt"

if [ -n "${AGL_EVAL_SCRIPT:-}" ]; then
    echo "$AGL_EVAL_SCRIPT" > /tmp/eval.sh
    chmod +x /tmp/eval.sh
    # Run eval script, capture output (allow failure — we just capture the log).
    bash /tmp/eval.sh > "$TEST_OUTPUT_FILE" 2>&1 || true
    echo "Test output: $(wc -c < "$TEST_OUTPUT_FILE") bytes"
else
    echo "WARNING: No AGL_EVAL_SCRIPT set, skipping evaluation"
    echo "No eval script provided" > "$TEST_OUTPUT_FILE"
fi

# ── Phase 4: Grade + post reward ─────────────────────────────────
echo "--- Phase 4: Grade + Reward ---"

# Install swebench for grading (~0.9MB, pure Python).
python3 -m pip install swebench -q 2>/dev/null || python3 -m pip install swebench 2>&1 | tail -1

python3 /agl/agents/grade.py "$TEST_OUTPUT_FILE"

# ── Phase 5: Archive outputs + post agent_output event ───────────
echo "--- Phase 5: Archive ---"

# Copy local outputs to shared volume.
mkdir -p "$ARTIFACT_DIR"
cp -r "$OUTPUT_DIR"/. "$ARTIFACT_DIR"/ 2>/dev/null \
    && echo "Archived outputs to $ARTIFACT_DIR" \
    || echo "WARNING: Failed to archive outputs (volume not mounted?)"

# Post agent_output event with patch summary and artifact path.
if [ -n "${AGL_EVENT_URL:-}" ]; then
    INSTANCE_ID=$(echo "${AGL_EVAL_META:-{}}" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('instance_id',''))" 2>/dev/null || echo "")
    REL_PATH="${ROLLOUT_ID}/${ATTEMPT_ID}"

    # Read patch content for the event (may be large, pipe via stdin).
    PATCH_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" < "$PATCH_FILE")

    python3 -c "
import json, os, sys, urllib.request

payload = json.dumps({
    'event_type': 'agent_output',
    'data': {
        'patch': json.loads(sys.argv[1]),
        'instance_id': sys.argv[2],
        'patch_size': int(sys.argv[3]),
        'artifact_path': sys.argv[4],
    },
}).encode()

req = urllib.request.Request(
    os.environ['AGL_EVENT_URL'],
    data=payload,
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.environ.get(\"AGL_KEY\", os.environ.get(\"OPENAI_API_KEY\", \"\"))}',
    },
    method='POST',
)
try:
    urllib.request.urlopen(req, timeout=30)
except Exception as e:
    print(f'WARNING: Failed to post agent_output event: {e}')
" "$PATCH_JSON" "$INSTANCE_ID" "$PATCH_SIZE" "$REL_PATH" \
    || echo "WARNING: Failed to post agent_output event"
fi

echo "=== Entrypoint complete ==="
