# Vercel AI SDK agent

This example connects a minimal TypeScript agent built with Vercel AI SDK to Agent Lightning v1. It uses the Gateway's OpenAI-compatible Chat Completions endpoint, executes a bounded arithmetic tool loop, and reports a GSM8K reward.

Agent Lightning v1 and Vercel AI SDK 6 are separate projects with independent version numbers. This example requires Node.js 22 or newer and pins its JavaScript dependencies in `package-lock.json`.

## Build and test

From the repository root:

```bash
cd examples/ai_sdk
npm ci
npm test
cd ../..
```

The tests run against a local HTTP stub. They do not require an LLM, GPU, or Agent Lightning server.

## Use with the GSM8K trainer

Build the TypeScript agent, prepare the GSM8K parquet files as described in the [GSM8K guide](../../docs/55-example-gsm8k.md), then use the existing Linux launcher with the agent class and tool-calling overrides below. The default Qwen model needs vLLM's automatic tool choice and Hermes parser, as in the Calc-X example:

```bash
cd examples/gsm8k
bash run_local.sh \
  --train-file /path/to/train.parquet \
  --val-file /path/to/test.parquet \
  agentlightning.local.agent_class=examples.ai_sdk.ai_sdk_agent.AISDKAgent \
  actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
  actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes
```

`run_local.sh` starts Ray, `agl-server`, and `agl-controller` before invoking the GSM8K trainer, then cleans them up on exit. The trainer's existing local configuration maps each rollout's `input.question` and `input.answer` to `QUESTION` and `ANSWER`. The local controller also injects `AGL_OPENAI_BASE_URL`, `AGL_EVENT_URL`, and `AGL_KEY`. `AISDKAgent` starts the compiled Node.js entrypoint without a shell and passes through that environment.

Set `AI_SDK_NODE` if `node` is not on the controller's `PATH`. The local controller and the training stack require Linux; rollout-driven VERL training also requires the GPU environment documented by Agent Lightning. This example's training path has not been validated on the Windows development machine used for its HTTP-stub tests.
