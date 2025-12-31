/**
 * WebShop Agent
 *
 * A ToolLoopAgent that navigates the real WebShop server environment
 * to find and purchase products matching user instructions.
 */

import { createOpenAI } from '@ai-sdk/openai';
import { ToolLoopAgent, InferAgentUIMessage, stepCountIs } from 'ai';
import { createWebShopTools, WebShopTools } from '@/tools';

/**
 * Options for creating a WebShop agent.
 */
export interface WebShopAgentOptions {
  /** Model ID to use (defaults to WEBSHOP_MODEL env var or 'gpt-4o-mini') */
  modelId?: string;
  /** AI SDK telemetry settings for OpenTelemetry tracing */
  telemetry?: {
    isEnabled: boolean;
    tracer?: unknown;
    recordInputs?: boolean;
    recordOutputs?: boolean;
  };
}

/**
 * Create an OpenAI-compatible client.
 * Supports configurable base URL for vLLM, LLMProxy, or other OpenAI-compatible endpoints.
 */
const openaiCompatible = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY ?? 'dummy',
  baseURL: process.env.OPENAI_API_BASE,
});

/**
 * System prompt for the WebShop agent.
 * Simplified for the real WebShop action grammar (search[...], click[...]).
 */
export const WEBSHOP_SYSTEM_PROMPT = `You are a shopping assistant agent navigating the WebShop environment.

## Available Actions
You can take these actions:
- **search**: issues search[query] to find products
- **click**: issues click[element] to interact with the page
- **buy**: convenience for click[Buy Now] to complete purchase

## Strategy
1. Search for products matching the main attributes (type, color, material)
2. Click on promising results to view product details
3. On product pages, click option values to configure (size, color, etc.)
4. Click "Buy Now" when all required options are selected

## Important
- Read the observation text carefully after each action
- The environment may return HTML-like text or structured observations
- Keep within the 15-step limit
- Consider price constraints if mentioned in the task`;

/**
 * Create a WebShop agent bound to a session.
 */
export function createWebShopAgent(
  sessionId: string,
  options?: WebShopAgentOptions
) {
  const tools = createWebShopTools(sessionId);
  const modelId =
    options?.modelId ?? process.env.WEBSHOP_MODEL ?? 'gpt-4o-mini';

  // Build agent configuration
  const agentConfig: Record<string, unknown> = {
    model: openaiCompatible(modelId),
    instructions: WEBSHOP_SYSTEM_PROMPT,
    tools,
    stopWhen: stepCountIs(15),
  };

  // Add telemetry if configured
  if (options?.telemetry) {
    agentConfig.experimental_telemetry = options.telemetry;
  }

  return new ToolLoopAgent(agentConfig as Parameters<typeof ToolLoopAgent>[0]);
}

/**
 * Type for agent UI messages.
 */
export type WebShopAgentUIMessage = InferAgentUIMessage<
  ReturnType<typeof createWebShopAgent>
>;

/**
 * Type for tool invocations in UI messages.
 */
export type WebShopUIToolInvocation = NonNullable<
  WebShopAgentUIMessage['parts'][number] extends infer P
    ? P extends { type: `tool-${string}` }
      ? P
      : never
    : never
>;
