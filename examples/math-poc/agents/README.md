# Example Agents

Reusable agent templates. Pure source code — no Dockerfile here.
Each PoC scenario builds its own agent image (see `examples/math-poc/`).

## qa_agent.py

Minimal QA agent: reads `AGL_TASK_INPUT`, makes one LLM call, prints the result.

- Uses `openai` SDK (built-in retry on 503)
- Does NOT import agl-lite — proves language-agnostic contract
- Supports `CRASH_ON_FIRST=1` env var for K8s retry testing

## react_agent.py

Placeholder for future multi-turn agent with tool-use loop.
