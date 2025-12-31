#!/usr/bin/env npx tsx
/**
 * Headless WebShop Rollout Runner
 *
 * Continuously dequeues rollouts from Agent Lightning Store and executes them
 * without the UI. Reports results back via REST API.
 *
 * Usage:
 *   npx tsx scripts/headless-runner.ts --worker-id runner-1
 *   npx tsx scripts/headless-runner.ts --once  # Run single task and exit
 *
 * Environment Variables:
 *   AGENT_LIGHTNING_STORE_URL - Agent Lightning Store URL (required)
 *   WEBSHOP_URL - WebShop server URL (default: http://localhost:3000)
 *   OPENAI_API_KEY - OpenAI API key
 *   OPENAI_API_BASE - OpenAI-compatible endpoint base URL
 *   WEBSHOP_MODEL - Model ID to use (default: gpt-4o-mini)
 *   AGENT_LIGHTNING_SERVICE_NAME - Service name for tracing
 */

import { createOpenAI } from '@ai-sdk/openai';
import { generateText } from 'ai';
import {
  AgentLightningStoreClient,
  createRolloutTracer,
  getOtlpEndpoint,
  makeAiSdkTelemetry,
  getProxyLLMBaseUrl,
  getMainLLM,
  isProxyLLM,
  type AttemptedRollout,
  type ProxyLLMResource,
  type LLMResource,
} from '../src/utils/agentlightning';
import { WebShopServerEnv } from '../src/environment/webshop-server';
import { WEBSHOP_SYSTEM_PROMPT } from '../src/agent/webshop-agent';

interface RunnerOptions {
  workerId: string;
  pollIntervalMs: number;
  once: boolean;
  maxSteps: number;
}

interface WebShopTask {
  task_id: string;
  instruction: string;
  target_attributes?: Record<string, unknown>;
}

/**
 * Parse action from model response text.
 * Looks for search[...] or click[...] patterns.
 */
function parseAction(text: string): string | null {
  // Look for search[...] pattern
  const searchMatch = text.match(/search\[([^\]]+)\]/i);
  if (searchMatch) return `search[${searchMatch[1]}]`;

  // Look for click[...] pattern
  const clickMatch = text.match(/click\[([^\]]+)\]/i);
  if (clickMatch) return `click[${clickMatch[1]}]`;

  // Look for buy now
  const buyMatch = text.match(/buy\s*now|buy\[\]/i);
  if (buyMatch) return 'click[Buy Now]';

  return null;
}

/**
 * Execute a single rollout.
 */
async function runRollout(
  rollout: AttemptedRollout,
  storeClient: AgentLightningStoreClient,
  options: RunnerOptions,
  llmResource: ProxyLLMResource | LLMResource | null
): Promise<void> {
  const { rollout_id, attempt } = rollout;
  const { attempt_id } = attempt;
  const task = rollout.input as WebShopTask;

  console.log(`[TASK] ${options.workerId} | rollout=${rollout_id.slice(0, 8)}... | ${task.instruction.slice(0, 60)}...`);

  // Initialize WebShop environment
  const webshopUrl = process.env.WEBSHOP_URL ?? 'http://localhost:3000';
  const env = new WebShopServerEnv({ baseUrl: webshopUrl, timeoutMs: 30_000 });

  // Reset environment with task
  try {
    await env.reset({
      task: {
        taskId: task.task_id,
        instruction: task.instruction,
        targetAttributes: task.target_attributes ?? {},
      },
    });
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : String(error);
    console.error(
      `[${options.workerId}] Failed to reset environment:`,
      errorMessage
    );
    await storeClient.completeAttempt(rollout_id, attempt_id, {
      success: false,
      error: `Failed to reset environment: ${errorMessage}`,
    });
    return;
  }

  // Create OpenAI-compatible client
  // Priority: 1. LLM resource from Store (with rollout routing), 2. OPENAI_API_BASE env var
  let baseURL: string | undefined;
  let modelId = process.env.WEBSHOP_MODEL ?? 'gpt-4o-mini';

  if (llmResource) {
    // Use the LLM resource from the Store
    if (isProxyLLM(llmResource)) {
      // ProxyLLM: construct routed endpoint for proper trace attribution
      baseURL = getProxyLLMBaseUrl(llmResource, rollout_id, attempt_id);
    } else {
      // Regular LLM: use endpoint directly
      baseURL = llmResource.endpoint;
    }
    modelId = llmResource.model;
    console.log(`[${options.workerId}] Using LLM Proxy: ${baseURL} (model: ${modelId})`);
  } else if (process.env.OPENAI_API_BASE) {
    // Fallback to environment variable
    baseURL = process.env.OPENAI_API_BASE;
    console.log(`[${options.workerId}] Using OPENAI_API_BASE: ${baseURL}`);
  }

  const openai = createOpenAI({
    apiKey: process.env.OPENAI_API_KEY ?? 'dummy',
    ...(baseURL && { baseURL }),
  });

  // Create tracer for this rollout
  const otlpEndpoint = getOtlpEndpoint();
  const serviceName =
    process.env.AGENT_LIGHTNING_SERVICE_NAME ?? 'webshop-headless-runner';
  let provider: Awaited<ReturnType<typeof createRolloutTracer>>['provider'];
  let telemetry: ReturnType<typeof makeAiSdkTelemetry> | undefined;

  if (otlpEndpoint) {
    const result = createRolloutTracer({
      otlpEndpoint,
      serviceName,
      rolloutId: rollout_id,
      attemptId: attempt_id,
    });
    provider = result.provider;
    telemetry = makeAiSdkTelemetry(result.tracer);
  }

  try {
    // Execute agent loop
    let stepCount = 0;
    let done = false;
    let reward = 0;
    let currentObservation = env.getState().observation;

    while (!done && stepCount < options.maxSteps) {
      stepCount++;

      // Generate next action using the model
      const response = await generateText({
        model: openai(modelId),
        system: WEBSHOP_SYSTEM_PROMPT,
        prompt: `Task: ${task.instruction}\n\nCurrent observation:\n${currentObservation}\n\nWhat action should I take next? Respond with search[query] or click[element].`,
        experimental_telemetry: telemetry,
      });

      // Parse action from response
      const action = parseAction(response.text);
      if (!action) {
        console.log(
          `[${options.workerId}] Step ${stepCount}: Could not parse action from: ${response.text.slice(0, 100)}...`
        );
        // Try to extract any actionable text
        const fallbackAction = response.text.includes('search')
          ? 'search[product]'
          : response.text.includes('click')
            ? 'click[Buy Now]'
            : null;
        if (!fallbackAction) break;
      }

      // Execute action
      const result = await env.step(action!);
      currentObservation = result.observation;
      done = result.done;
      reward = result.reward ?? 0;

      // Compact step log for easy filtering with grep
      const resultIcon = done ? (reward > 0 ? '✓' : '✗') : '→';
      console.log(`[STEP] ${stepCount}. ${action} ${resultIcon}${done ? ` reward=${reward.toFixed(2)}` : ''}`);
    }

    // Flush traces BEFORE completing the attempt to avoid race condition
    // The coordinator queries for spans immediately when it sees the rollout is complete
    if (provider) {
      await provider.forceFlush();
    }

    // Report success
    await storeClient.completeAttempt(rollout_id, attempt_id, {
      success: reward > 0,
      reward,
    });

    // Summary log for training progress tracking
    const status = reward > 0 ? 'SUCCESS' : 'FAIL';
    console.log(`[DONE] ${options.workerId} | reward=${reward.toFixed(2)} | steps=${stepCount} | ${status}`);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : String(error);
    console.error(
      `[${options.workerId}] Rollout ${rollout_id} failed:`,
      errorMessage
    );

    // Flush traces before reporting error too
    if (provider) {
      await provider.forceFlush().catch(() => {});
    }

    await storeClient.completeAttempt(rollout_id, attempt_id, {
      success: false,
      error: errorMessage,
    });
  }
}

/**
 * Main runner loop.
 */
async function runLoop(options: RunnerOptions): Promise<void> {
  const storeUrl = process.env.AGENT_LIGHTNING_STORE_URL;
  if (!storeUrl) {
    console.error('Error: AGENT_LIGHTNING_STORE_URL environment variable is not set');
    process.exit(1);
  }

  const storeClient = new AgentLightningStoreClient({ baseUrl: storeUrl });

  // Check if store is healthy
  const healthy = await storeClient.health();
  if (!healthy) {
    console.error(`Error: Agent Lightning Store at ${storeUrl} is not healthy`);
    process.exit(1);
  }

  console.log(`[${options.workerId}] Starting headless runner`);
  console.log(`[${options.workerId}] Store URL: ${storeUrl}`);
  console.log(`[${options.workerId}] Poll interval: ${options.pollIntervalMs}ms`);
  console.log(`[${options.workerId}] Max steps per task: ${options.maxSteps}`);

  // Fetch LLM resource from Store (published by VERL/LLM Proxy)
  // This allows dynamic discovery of the vLLM endpoint during training
  let llmResource: ProxyLLMResource | LLMResource | null = null;
  const resources = await storeClient.getLatestResources();
  if (resources) {
    llmResource = getMainLLM(resources);
    if (llmResource) {
      const resourceType = isProxyLLM(llmResource) ? 'ProxyLLM' : 'LLM';
      console.log(`[${options.workerId}] Found ${resourceType} resource: ${llmResource.model} @ ${llmResource.endpoint}`);
    }
  }
  if (!llmResource && process.env.OPENAI_API_BASE) {
    console.log(`[${options.workerId}] No LLM resource in Store, using OPENAI_API_BASE fallback`);
  } else if (!llmResource) {
    console.log(`[${options.workerId}] No LLM resource found, will use OpenAI default endpoint`);
  }

  while (true) {
    try {
      // Try to dequeue a rollout
      const rollouts = await storeClient.dequeueRollouts(1, options.workerId);

      if (rollouts.length > 0) {
        await runRollout(rollouts[0], storeClient, options, llmResource);

        if (options.once) {
          console.log(`[${options.workerId}] Single run mode, exiting`);
          break;
        }
      } else {
        // No work available, wait and poll again
        if (options.once) {
          console.log(`[${options.workerId}] No tasks available, exiting (single run mode)`);
          break;
        }
        await new Promise((resolve) =>
          setTimeout(resolve, options.pollIntervalMs)
        );
      }
    } catch (error) {
      console.error(`[${options.workerId}] Error:`, error);
      await new Promise((resolve) =>
        setTimeout(resolve, options.pollIntervalMs)
      );
    }
  }
}

// Parse CLI arguments
function parseArgs(): RunnerOptions {
  const args = process.argv.slice(2);

  const getArg = (flag: string): string | undefined => {
    const index = args.findIndex((a) => a === flag);
    return index !== -1 && index + 1 < args.length
      ? args[index + 1]
      : undefined;
  };

  return {
    workerId: getArg('--worker-id') ?? `runner-${Date.now()}`,
    pollIntervalMs: parseInt(getArg('--poll-interval') ?? '1000'),
    once: args.includes('--once'),
    maxSteps: parseInt(getArg('--max-steps') ?? '15'),
  };
}

// Run
const options = parseArgs();
runLoop(options).catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
