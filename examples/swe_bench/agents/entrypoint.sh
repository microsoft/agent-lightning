#!/bin/bash
# Shared entrypoint for SWE-bench agent containers.
#
# Runs inside the official SWE-bench Docker image at /testbed.
# Dispatches to the agent specified by AGL_CODING_AGENT env var,
# then runs eval_script, grades using official swebench tools,
# and posts a small reward event.
#
# Expected env vars (set by hook via on_enqueue):
#   AGL_TASK_INPUT      — problem statement
#   AGL_EVAL_SCRIPT     — bash script to run tests
#   AGL_EVAL_META       — JSON with FAIL_TO_PASS, PASS_TO_PASS, instance_id
#   AGL_CODING_AGENT    — agent name (e.g., "claude_code")
#   AGL_EVENT_URL       — URL to post events (set by controller)
#   AGL_ROLLOUT_ID      — rollout ID (set by controller)
#   OPENAI_BASE_URL     — LLM proxy URL (set by controller)
#   OPENAI_API_KEY      — API key (set by controller)

set -euo pipefail

AGENT_DIR="/agl/agents"
AGENT_NAME="${AGL_CODING_AGENT:-claude_code}"
ARTIFACT_DIR="/data/artifacts/${AGL_ROLLOUT_ID:-unknown}"

echo "=== SWE-bench Agent Entrypoint ==="
echo "  Agent: $AGENT_NAME"
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

PATCH=$(git -c core.fileMode=false diff HEAD 2>/dev/null || echo "")
PATCH_SIZE=${#PATCH}
echo "Patch size: $PATCH_SIZE bytes"

# Post agent_output event with patch.
if [ -n "$AGL_EVENT_URL" ]; then
    PATCH_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$PATCH")
    INSTANCE_ID=$(echo "$AGL_EVAL_META" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('instance_id',''))" 2>/dev/null || echo "")

    curl -sf -X POST "$AGL_EVENT_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${AGL_KEY:-${OPENAI_API_KEY:-}}" \
        -d "{\"event_type\":\"agent_output\",\"data\":{\"patch\":${PATCH_JSON},\"instance_id\":\"${INSTANCE_ID}\",\"patch_size\":${PATCH_SIZE}}}" \
        || echo "WARNING: Failed to post agent_output event"
fi

# ── Phase 3: Run eval_script ─────────────────────────────────────
echo "--- Phase 3: Evaluate ---"

TEST_OUTPUT_FILE="/tmp/test_output.txt"

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

# ── Phase 4: Grade using official swebench tools ─────────────────
echo "--- Phase 4: Grade ---"

REWARD_VALUE="0.0"
RESOLVED="false"
REASON="unknown"

if [ -f "$TEST_OUTPUT_FILE" ] && [ -n "${AGL_EVAL_META:-}" ]; then
    # Install swebench if not present (~0.9MB, pure Python).
    pip install swebench -q 2>/dev/null || pip install swebench 2>&1 | tail -1

    # Grade: parse test output using official get_eval_report().
    GRADE_RESULT=$(python3 -c "
import json, os, sys
from types import SimpleNamespace

eval_meta = json.loads(os.environ['AGL_EVAL_META'])
instance_id = eval_meta['instance_id']

# Build a minimal test_spec-like object with the fields get_eval_report needs.
test_spec = SimpleNamespace(
    instance_id=instance_id,
    repo=eval_meta.get('repo', ''),
    version=eval_meta.get('version', ''),
    FAIL_TO_PASS=eval_meta['FAIL_TO_PASS'],
    PASS_TO_PASS=eval_meta['PASS_TO_PASS'],
)

from swebench.harness.grading import get_eval_report

prediction = {
    'instance_id': instance_id,
    'model_patch': '',
    'model_name_or_path': 'agl-lite',
}

try:
    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path='$TEST_OUTPUT_FILE',
        include_tests_status=True,
    )
    resolved = report.get(instance_id, {}).get('resolved', False)
    reward = 1.0 if resolved else 0.0
    reason = 'resolved' if resolved else 'not resolved'
except Exception as e:
    reward = 0.0
    resolved = False
    reason = f'grading error: {e}'

print(json.dumps({'reward': reward, 'resolved': resolved, 'reason': reason}))
" 2>/dev/null || echo '{"reward":0.0,"resolved":false,"reason":"grade script failed"}')

    REWARD_VALUE=$(echo "$GRADE_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['reward'])")
    RESOLVED=$(echo "$GRADE_RESULT" | python3 -c "import json,sys; r=json.loads(sys.stdin.read())['resolved']; print('true' if r else 'false')")
    REASON=$(echo "$GRADE_RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['reason'])")

    echo "Grade: reward=$REWARD_VALUE resolved=$RESOLVED reason=$REASON"
else
    REASON="no test output or eval meta"
    echo "Skipping grading: $REASON"
fi

# ── Phase 5: Post reward event ───────────────────────────────────
echo "--- Phase 5: Post reward ---"

if [ -n "$AGL_EVENT_URL" ]; then
    INSTANCE_ID=$(echo "$AGL_EVAL_META" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('instance_id',''))" 2>/dev/null || echo "")

    curl -sf -X POST "$AGL_EVENT_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${AGL_KEY:-${OPENAI_API_KEY:-}}" \
        -d "{\"event_type\":\"reward\",\"data\":{\"value\":${REWARD_VALUE},\"resolved\":${RESOLVED},\"instance_id\":\"${INSTANCE_ID}\",\"patch_size\":${PATCH_SIZE},\"reason\":\"${REASON}\"}}" \
        || echo "WARNING: Failed to post reward event"
fi

# ── Phase 6: Archive test log to shared volume (debugging only) ──
echo "--- Phase 6: Archive ---"

if [ -f "$TEST_OUTPUT_FILE" ]; then
    mkdir -p "$ARTIFACT_DIR"
    cp "$TEST_OUTPUT_FILE" "$ARTIFACT_DIR/test_output.txt" 2>/dev/null \
        && echo "Archived test output to $ARTIFACT_DIR/test_output.txt" \
        || echo "WARNING: Failed to archive test output (volume not mounted?)"
fi

echo "=== Entrypoint complete ==="
