/**
 * WebShop Tools
 *
 * AI SDK tool definitions for the WebShop server environment with streaming support.
 * Tools use generator functions to yield intermediate states for real-time UI updates.
 *
 * Note: These tools are used by the UI for debugging/testing. Training is handled
 * by the headless runner which has its own execution loop.
 */

import { tool } from 'ai';
import { z } from 'zod';
import {
  getEnvironment,
  getRecorder,
  isSessionInitialized,
  markSessionInitialized,
  getTask,
} from '@/environment/session-manager';

/**
 * Search tool output state
 */
export type SearchToolOutput =
  | { state: 'loading' }
  | {
      state: 'ready';
      success: boolean;
      query: string;
      observation: string;
      done: boolean;
      reward?: number;
    };

/**
 * Click tool output state
 */
export type ClickToolOutput =
  | { state: 'loading' }
  | {
      state: 'ready';
      success: boolean;
      element: string;
      observation: string;
      done: boolean;
      reward?: number;
    };

/**
 * Buy tool output state
 */
export type BuyToolOutput =
  | { state: 'loading' }
  | {
      state: 'ready';
      success: boolean;
      observation: string;
      done: boolean;
      reward: number;
    };

/**
 * Create WebShop tools bound to a session.
 */
export function createWebShopTools(sessionId: string) {
  return {
    /**
     * Search for products in the WebShop.
     * Maps to search[query] in the WebShop action grammar.
     */
    search: tool({
      description:
        'Search for products in the store. This maps to search[query] in WebShop. ' +
        'Example queries: "red cotton t-shirt", "running shorts", "fleece hoodie"',
      inputSchema: z.object({
        query: z
          .string()
          .describe(
            'The search query to find products (e.g., "men red cotton t-shirt")'
          ),
      }),
      execute: async function* ({ query }): AsyncGenerator<SearchToolOutput> {
        // Yield loading state for UI
        yield { state: 'loading' };

        const env = getEnvironment(sessionId);
        if (!env) {
          yield {
            state: 'ready',
            success: false,
            query,
            observation: 'Error: Session not found.',
            done: false,
          };
          return;
        }

        // Initialize session if needed
        if (!isSessionInitialized(sessionId)) {
          const task = getTask(sessionId);
          await env.reset({ task });
          markSessionInitialized(sessionId);
        }

        const result = await env.search(query);

        // Record step for trajectory
        const recorder = getRecorder(sessionId);
        if (recorder) {
          recorder.recordStep({
            toolCalls: [{ toolName: 'search', args: { query } }],
            toolResults: [{ toolName: 'search', result: result.observation }],
            text: '',
          });
        }

        // Yield final state
        yield {
          state: 'ready',
          success: result.success,
          query,
          observation: result.observation,
          done: result.done,
          reward: result.reward,
        };
      },
    }),

    /**
     * Click on an element in the current page.
     * Maps to click[element] in the WebShop action grammar.
     */
    click: tool({
      description:
        'Click on an element in the current page. This maps to click[element] in WebShop. ' +
        'Can be used to: select a product from search results, ' +
        'select product options like size or color, ' +
        'go back to search, or complete purchase with "Buy Now".',
      inputSchema: z.object({
        element: z
          .string()
          .describe(
            'The element to click (product name, option value, "Back to Search", or "Buy Now")'
          ),
      }),
      execute: async function* ({ element }): AsyncGenerator<ClickToolOutput> {
        // Yield loading state for UI
        yield { state: 'loading' };

        const env = getEnvironment(sessionId);
        if (!env) {
          yield {
            state: 'ready',
            success: false,
            element,
            observation: 'Error: Session not found.',
            done: false,
          };
          return;
        }

        // Initialize session if needed
        if (!isSessionInitialized(sessionId)) {
          const task = getTask(sessionId);
          await env.reset({ task });
          markSessionInitialized(sessionId);
        }

        const result = await env.click(element);

        // Record step for trajectory
        const recorder = getRecorder(sessionId);
        if (recorder) {
          recorder.recordStep({
            toolCalls: [{ toolName: 'click', args: { element } }],
            toolResults: [{ toolName: 'click', result: result.observation }],
            text: '',
          });
        }

        yield {
          state: 'ready',
          success: result.success,
          element,
          observation: result.observation,
          done: result.done,
          reward: result.reward,
        };
      },
    }),

    /**
     * Buy the currently viewed product.
     * Convenience tool that maps to click[Buy Now].
     */
    buy: tool({
      description:
        'Purchase the currently viewed product. This is equivalent to click[Buy Now]. ' +
        'Use this when you are on a product page and have selected all required options.',
      inputSchema: z.object({}),
      execute: async function* (): AsyncGenerator<BuyToolOutput> {
        // Yield loading state for UI
        yield { state: 'loading' };

        const env = getEnvironment(sessionId);
        if (!env) {
          yield {
            state: 'ready',
            success: false,
            observation: 'Error: Session not found.',
            done: false,
            reward: 0,
          };
          return;
        }

        // Initialize session if needed
        if (!isSessionInitialized(sessionId)) {
          const task = getTask(sessionId);
          await env.reset({ task });
          markSessionInitialized(sessionId);
        }

        const result = await env.buy();

        // Record step for trajectory
        const recorder = getRecorder(sessionId);
        if (recorder) {
          recorder.recordStep({
            toolCalls: [{ toolName: 'buy', args: {} }],
            toolResults: [{ toolName: 'buy', result: result.observation }],
            text: '',
          });
        }

        yield {
          state: 'ready',
          success: result.success,
          observation: result.observation,
          done: result.done,
          reward: result.reward ?? 0,
        };
      },
    }),
  };
}

export type WebShopTools = ReturnType<typeof createWebShopTools>;
