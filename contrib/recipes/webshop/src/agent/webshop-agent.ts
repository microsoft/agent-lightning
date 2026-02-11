// Copyright (c) Microsoft. All rights reserved.

/**
 * WebShop Agent
 *
 * A ToolLoopAgent that navigates the real WebShop server environment
 * to find and purchase products matching user instructions.
 *
 * NOTE: This module is for the Next.js UI and requires @/tools which uses
 * Next.js-specific features. For headless/CLI usage, use prompts.ts directly.
 */

import { createOpenAI } from '@ai-sdk/openai';
import { ToolLoopAgent, InferAgentUIMessage, stepCountIs } from 'ai';
import { createWebShopTools, WebShopTools } from '@/tools';
import { WEBSHOP_SYSTEM_PROMPT } from './prompts';

// Re-export for backwards compatibility
export { WEBSHOP_SYSTEM_PROMPT };

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
