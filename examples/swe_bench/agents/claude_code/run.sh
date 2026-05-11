#!/bin/bash
# Claude Code agent — run Claude Code on the SWE-bench problem.
#
# Uses the problem statement from AGL_TASK_INPUT.
# CLAUDE.md is copied to /testbed so Claude Code picks it up as project context.
#
# Key env vars (set by controller/hook):
#   AGL_TASK_INPUT       — problem statement
#   ANTHROPIC_BASE_URL   — LLM proxy URL (agl-lite gateway)
#   ANTHROPIC_AUTH_TOKEN  — alias for ANTHROPIC_API_KEY
#   OPENAI_BASE_URL      — also set by controller (Claude Code may use either)

set -euo pipefail

cd /testbed
export PATH="$HOME/.local/bin:$PATH"
export IS_SANDBOX=1

# Copy CLAUDE.md project instructions if available.
if [ -f /agl/agents/claude_code--CLAUDE.md ]; then
    cp /agl/agents/claude_code--CLAUDE.md /testbed/CLAUDE.md
fi

PROBLEM="${AGL_TASK_INPUT:-}"
if [ -z "$PROBLEM" ]; then
    echo "ERROR: AGL_TASK_INPUT not set"
    exit 1
fi

MAX_TURNS="${AGL_MAX_TURNS:-100}"

echo "=== Claude Code Agent ==="
echo "Max turns: $MAX_TURNS"
echo "Problem: ${PROBLEM:0:200}..."

# Use ANTHROPIC_AUTH_TOKEN if ANTHROPIC_API_KEY not set.
# The controller sets both OPENAI_API_KEY and ANTHROPIC_API_KEY from AGL_KEY.
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}"

# Write prompt to a file to avoid shell escaping issues (heredoc approach from Agent Lightning).
cat > /tmp/cc_prompt.txt << 'CC_PROMPT'
You are given a code repository in the current directory (/testbed).
The bug description is:
CC_PROMPT
echo "$PROBLEM" >> /tmp/cc_prompt.txt
cat >> /tmp/cc_prompt.txt << 'CC_PROMPT2'
=================================================
This is an authorized SWE-bench benchmark running inside a disposable local
container. You may inspect and edit files in /testbed to repair this repository.

Your task:
(1) understand the bug report,
(2) locate the relevant implementation code,
(3) apply a minimal source-code fix,
(4) do not commit changes and do not leave new test files behind.

The harness will run the official evaluation after you finish. Work only inside
this repository checkout.
CC_PROMPT2

EXTRA_SYSTEM_PROMPT="You are an expert software engineer solving an authorized local SWE-bench bug fixing task. Editing files in /testbed is expected. Do not refuse because this is a local benchmark container."

# Run Claude Code.
claude -p "$(cat /tmp/cc_prompt.txt)" \
    --append-system-prompt "$EXTRA_SYSTEM_PROMPT" \
    --max-turns "$MAX_TURNS" \
    --dangerously-skip-permissions \
    --output-format json \
    --verbose \
    2>&1 || echo "WARNING: Claude Code exited with code $?"

echo "=== Claude Code complete ==="
