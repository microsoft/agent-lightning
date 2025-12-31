/**
 * Next.js OpenTelemetry Instrumentation
 *
 * Enables async context management for proper span parenting in server components.
 * This is required for AI SDK telemetry to work correctly.
 */

import { context } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';

export function register() {
  try {
    const contextManager = new AsyncLocalStorageContextManager();
    contextManager.enable();
    context.setGlobalContextManager(contextManager);
  } catch {
    // Context manager already set, ignore
  }
}
