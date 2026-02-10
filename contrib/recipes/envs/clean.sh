#!/bin/bash
set -e

echo "Stopping AgentLightning and simulation_agent..."
pkill -f AgentLightning || true
pkill -f simulation_agent || true

echo "Stopping Ray cluster..."
ray stop

echo "Killing VLLM::EngineCore processes..."
ps aux | grep VLLM::EngineCore | grep -v grep | awk '{print $2}' | xargs --no-run-if-empty kill -9

echo "✅ Cleanup complete."