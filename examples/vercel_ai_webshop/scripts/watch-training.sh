#!/bin/bash
# watch-training.sh - Launch tmux session for training visibility
#
# Creates a 3-pane layout showing:
#   - Top (60%): Agent decisions (runner logs filtered for [TASK], [STEP], [DONE])
#   - Middle (25%): Training progress (agl logs filtered for [PROGRESS])
#   - Bottom (15%): WebShop server logs
#
# Usage:
#   ./scripts/watch-training.sh
#   make watch

set -e

SESSION_NAME="webshop-training"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory for docker compose
cd "$PROJECT_DIR"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    echo "  Ubuntu/Debian: sudo apt install tmux"
    echo "  macOS: brew install tmux"
    exit 1
fi

# Check if docker compose is running
if ! docker compose ps --quiet 2>/dev/null | head -1 > /dev/null; then
    echo "Warning: No containers appear to be running."
    echo "Start the services first with: make dev"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "Starting training visibility session..."
echo "  - Top pane: Agent decisions ([TASK], [STEP], [DONE])"
echo "  - Middle pane: Training progress ([PROGRESS])"
echo "  - Bottom pane: WebShop server logs"
echo ""
echo "Press Ctrl+B then D to detach, or Ctrl+C in any pane to exit."
echo ""

# Create new session with first pane (agent decisions - top, largest)
tmux new-session -d -s "$SESSION_NAME" -n "training" \
    "docker compose logs -f runner runner-gpu 2>&1 | grep --line-buffered -E '^\S+\s+\| \[TASK\]|^\S+\s+\| \[STEP\]|^\S+\s+\| \[DONE\]|\[TASK\]|\[STEP\]|\[DONE\]' || echo 'Waiting for runner logs...'; read"

# Split horizontally for training progress (middle pane - 40% of remaining)
tmux split-window -v -t "$SESSION_NAME" -p 40 \
    "docker compose logs -f agl-server-dev agl-server-gpu 2>&1 | grep --line-buffered -E '\[PROGRESS\]' || echo 'Waiting for coordinator logs...'; read"

# Split again for webshop server logs (bottom pane - 30% of remaining)
tmux split-window -v -t "$SESSION_NAME" -p 30 \
    "docker compose logs -f webshop 2>&1 || echo 'Waiting for webshop logs...'; read"

# Add pane titles (requires tmux 2.6+)
tmux select-pane -t "$SESSION_NAME:0.0" -T "Agent Decisions"
tmux select-pane -t "$SESSION_NAME:0.1" -T "Training Progress"
tmux select-pane -t "$SESSION_NAME:0.2" -T "WebShop Server"

# Enable pane border status to show titles
tmux set-option -t "$SESSION_NAME" pane-border-status top 2>/dev/null || true
tmux set-option -t "$SESSION_NAME" pane-border-format " #{pane_title} " 2>/dev/null || true

# Select the top pane as active
tmux select-pane -t "$SESSION_NAME:0.0"

# Attach to the session
tmux attach-session -t "$SESSION_NAME"
