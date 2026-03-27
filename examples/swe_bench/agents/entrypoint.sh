#!/bin/bash
# Shared entrypoint for SWE-bench agent containers.
#
# Runs inside the official SWE-bench Docker image at /testbed.
# Dispatches to the agent specified by AGL_CODING_AGENT env var,
# then runs eval_script and posts artifacts.
#
# Expected env vars (set by hook via on_enqueue):
#   AGL_TASK_INPUT      — problem statement
#   AGL_EVAL_SCRIPT     — bash script to run tests
#   AGL_EVAL_META       — JSON with FAIL_TO_PASS, PASS_TO_PASS, instance_id
#   AGL_CODING_AGENT    — agent name (e.g., "mini_swe_agent", "claude_code")
#   AGL_EVENT_URL       — URL to post events (set by controller)
#   OPENAI_BASE_URL     — LLM proxy URL (set by controller)
#   OPENAI_API_KEY      — API key (set by controller)

set -euo pipefail

AGENT_DIR="/agl/agents"
AGENT_NAME="${AGL_CODING_AGENT:-claude_code}"

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
    # Escape patch for JSON.
    PATCH_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$PATCH")
    INSTANCE_ID=$(echo "$AGL_EVAL_META" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('instance_id',''))" 2>/dev/null || echo "")

    curl -sf -X POST "$AGL_EVENT_URL" \
        -H "Content-Type: application/json" \
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

# ── Phase 4: Post test output as artifact ────────────────────────
echo "--- Phase 4: Post artifacts ---"

if [ -n "$AGL_EVENT_URL" ] && [ -f "$TEST_OUTPUT_FILE" ]; then
    TEST_CONTENT=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" < "$TEST_OUTPUT_FILE")

    curl -sf -X POST "$AGL_EVENT_URL" \
        -H "Content-Type: application/json" \
        -d "{\"event_type\":\"artifact\",\"data\":{\"filename\":\"test_output.txt\",\"content\":${TEST_CONTENT}}}" \
        || echo "WARNING: Failed to post test_output artifact"
fi

echo "=== Entrypoint complete ==="
