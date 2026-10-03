# Vercel AI SDK agent

| Runtime | Model API | Controller | Training |
| --- | --- | --- | --- |
| Node.js 22+ | Chat Completions | Local | Reuses the GSM8K sync trainer |

This example shows how to run a TypeScript agent built with Vercel AI SDK through Agent Lightning v1. The agent uses `generateText` with a bounded arithmetic tool loop, sends non-streaming requests through the rollout-specific OpenAI-compatible proxy, and reports a GSM8K reward to the rollout event endpoint.

Agent Lightning v1 and Vercel AI SDK 6 have independent version numbers. The complete source, dependency pins, build steps, HTTP-stub tests, and GSM8K trainer override are in the [example README](https://github.com/microsoft/agent-lightning/tree/main/examples/ai_sdk).

The automated tests do not require an LLM or GPU. The full Agent Lightning training path requires Linux and the documented VERL GPU environment; it has not been validated on the Windows machine used to develop this example.
