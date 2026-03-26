#!/bin/bash
# Mini SWE Agent — lightweight Python agent for testing and OSS models.
# Uses the OpenAI API (via agl-lite gateway) to interact with the codebase.
#
# This is a minimal agent that:
# 1. Reads the problem statement
# 2. Explores the repository structure
# 3. Makes a single attempt at fixing the issue

set -euo pipefail

# Ensure we have the OpenAI client
pip install openai -q 2>/dev/null || true

python3 /agl/agents/mini_swe_agent/agent.py
