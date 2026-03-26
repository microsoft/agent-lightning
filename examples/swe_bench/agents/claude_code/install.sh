#!/bin/bash
# Claude Code agent — install the Claude CLI.
#
# Installs via the official installer and configures the environment.
# Uses ANTHROPIC_BASE_URL (set by controller to agl-lite gateway) for API routing.

set -euo pipefail

echo "Installing Claude Code CLI..."

# Install Claude CLI via official installer.
curl -fsSL https://claude.ai/install.sh | bash 2>/dev/null || {
    echo "WARNING: Official installer failed. Trying npm..."
    npm install -g @anthropic-ai/claude-code 2>/dev/null || {
        echo "ERROR: Could not install Claude CLI"
        exit 1
    }
}

# Make claude available in PATH.
export PATH="$HOME/.local/bin:$PATH"

# Configure Claude settings directory.
CLAUDE_CONFIG_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_CONFIG_DIR"

# Disable sandbox mode warning (we're already in a container).
export IS_SANDBOX=1

# Set up hook logging (captures tool use events for debugging).
cp /agl/agents/claude_code/handle_hook.sh /tmp/handle_hook.sh
chmod +x /tmp/handle_hook.sh

# Write Claude settings with hook config.
cat > "$CLAUDE_CONFIG_DIR/settings.json" << 'SETTINGS_EOF'
{
  "hooks": {
    "PreToolUse": [{"hooks": [{"type": "command", "command": "/tmp/handle_hook.sh"}]}],
    "PostToolUse": [{"hooks": [{"type": "command", "command": "/tmp/handle_hook.sh"}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "/tmp/handle_hook.sh"}]}]
  }
}
SETTINGS_EOF

echo "Claude CLI installed and configured."
