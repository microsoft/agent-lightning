#!/bin/bash
# Claude Code agent — install the Claude CLI.
# Requires ANTHROPIC_API_KEY or OPENAI_API_KEY for authentication.

set -euo pipefail

echo "Installing Claude Code CLI..."

# Install via official script.
curl -fsSL https://claude.ai/install.sh | sh 2>/dev/null || {
    echo "WARNING: Claude CLI install failed. Falling back to npm."
    npm install -g @anthropic-ai/claude-code 2>/dev/null || {
        echo "ERROR: Could not install Claude CLI"
        exit 1
    }
}

echo "Claude CLI installed."
