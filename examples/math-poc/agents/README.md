# Math PoC Agents

## qa_agent.py

Minimal QA agent for math problems:

1. Reads `AGL_TASK_INPUT` (JSON dict with `question` field)
2. Builds prompt: "You're a helpful math assistant... put answer in `\boxed{answer}`"
3. Calls LLM via `OPENAI_BASE_URL` (openai SDK, auto-retry on 503)
4. Parses `\boxed{answer}` from response
5. Posts `agent_output` event to `AGL_EVENT_URL` with extracted answer

Does NOT import agl-lite — proves language-agnostic contract.

### Environment variables

| Var | Source | Purpose |
|-----|--------|---------|
| `AGL_TASK_INPUT` | Controller | JSON dict with `question` field |
| `OPENAI_BASE_URL` | Controller | Points to agl-lite gateway |
| `OPENAI_API_KEY` | Controller (from Secret) | Auth key for gateway |
| `AGL_EVENT_URL` | Controller | URL to post agent_output events |
| `CRASH_ON_FIRST` | Optional | If "1", crash on first attempt (retry testing) |

### Reserved event types

| Type | Producer | Data |
|------|----------|------|
| `model_request` | Gateway (auto) | `{request, response, server}` |
| `agent_output` | Agent | `{answer, raw_response}` |
| `reward` | Algorithm | `{value, ground_truth, agent_answer}` |
