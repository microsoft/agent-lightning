/**
 * ProxyLLM Utilities
 *
 * Utilities for working with ProxyLLM resources from Agent Lightning.
 * These match the behavior of Python's ProxyLLM.get_base_url() method.
 */

import type { ProxyLLMResource, LLMResource, ResourcesUpdate } from './types';

/**
 * Construct the base URL for a ProxyLLM with rollout/attempt routing.
 *
 * This matches the Python implementation in agentlightning/types/resources.py:
 * ProxyLLM.get_base_url(rollout_id, attempt_id)
 *
 * The returned endpoint is:
 *   {endpoint}/rollout/{rollout_id}/attempt/{attempt_id}/v1
 *
 * @param resource - The ProxyLLM resource containing the base endpoint
 * @param rolloutId - Rollout identifier for span attribution
 * @param attemptId - Attempt identifier for span attribution
 * @returns Fully qualified endpoint including rollout metadata
 */
export function getProxyLLMBaseUrl(
  resource: ProxyLLMResource | LLMResource,
  rolloutId: string,
  attemptId: string
): string {
  let prefix = resource.endpoint;

  // Normalize: remove trailing slash
  if (prefix.endsWith('/')) {
    prefix = prefix.slice(0, -1);
  }

  // Handle /v1 suffix
  let hasV1 = false;
  if (prefix.endsWith('/v1')) {
    prefix = prefix.slice(0, -3);
    hasV1 = true;
  }

  // Append rollout/attempt routing
  prefix = `${prefix}/rollout/${rolloutId}/attempt/${attemptId}`;
  if (hasV1) {
    prefix += '/v1';
  }

  return prefix;
}

/**
 * Extract the main LLM resource from a ResourcesUpdate.
 *
 * VERL publishes resources with a 'main_llm' key that typically contains
 * a ProxyLLM resource pointing to the LLM Proxy endpoint.
 *
 * @param resources - The resources update from the Store
 * @returns The main LLM resource, or null if not found
 */
export function getMainLLM(
  resources: ResourcesUpdate
): ProxyLLMResource | LLMResource | null {
  const mainLlm = resources.resources['main_llm'];

  if (!mainLlm) {
    return null;
  }

  if (mainLlm.resource_type === 'proxy_llm' || mainLlm.resource_type === 'llm') {
    return mainLlm as ProxyLLMResource | LLMResource;
  }

  return null;
}

/**
 * Check if a resource is a ProxyLLM (supports rollout/attempt routing).
 */
export function isProxyLLM(
  resource: ProxyLLMResource | LLMResource
): resource is ProxyLLMResource {
  return resource.resource_type === 'proxy_llm';
}
