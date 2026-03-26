#!/bin/bash
# Claude Code agent — run Claude Code on the SWE-bench problem.
#
# Uses the problem statement from AGL_TASK_INPUT.
# CLAUDE.md is copied from the mounted ConfigMap if available.

set -euo pipefail

cd /testbed

# Copy CLAUDE.md instructions if available.
if [ -f /agl/agents/claude_code/CLAUDE.md ]; then
    cp /agl/agents/claude_code/CLAUDE.md /testbed/CLAUDE.md
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

# Run Claude Code with the problem statement.
claude -p "$PROBLEM" \
    --output-format stream-json \
    --max-turns "$MAX_TURNS" \
    --allowedTools "Bash(command)" "Edit(file_path, old_string, new_string)" "Read(file_path)" \
    2>&1 || echo "WARNING: Claude Code exited with code $?"

echo "=== Claude Code complete ==="
