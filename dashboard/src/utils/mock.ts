// Copyright (c) Microsoft. All rights reserved.

/**
 * Mock API handlers for testing and Storybook.
 *
 * This module provides utilities for creating MSW (Mock Service Worker) handlers
 * that simulate the Agent Lightning Store API. The implementation is based on the
 * actual Python server in agentlightning/store/client_server.py.
 * The mock covers a useful subset of the server's behavior; when functionality
 * is missing compared to the Python implementation it is an intentional tradeoff
 * aimed at keeping the mocks light for UI tests.
 *
 * @module mock
 */

import { delay, http, HttpResponse } from 'msw';
import type { Attempt, Resources, Rollout, Span, Worker } from '@/types';
import { snakeCaseKeys } from './format';

/**
 * Parse a numeric query parameter with a default value.
 * Returns the parsed number or the default if parsing fails.
 */
export function parseNumberParam(params: URLSearchParams, key: string, defaultValue: number): number {
  const raw = params.get(key);
  if (raw == null) {
    return defaultValue;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    return defaultValue;
  }
  return value;
}

/**
 * Filter rollouts based on query parameters.
 * Supports: status_in, mode_in, rollout_id_contains
 */
export function filterRolloutsForParams(rollouts: Rollout[], params: URLSearchParams): Rollout[] {
  const statusFilters = params.getAll('status_in');
  const modeFilters = params.getAll('mode_in');
  const rolloutIdContains = params.get('rollout_id_contains');
  const filterLogic = params.get('filter_logic') === 'or' ? 'or' : 'and';

  return rollouts.filter((rollout) => {
    const checks: boolean[] = [];
    if (statusFilters.length > 0) {
      checks.push(statusFilters.includes(rollout.status));
    }
    if (modeFilters.length > 0) {
      checks.push(rollout.mode != null && modeFilters.includes(rollout.mode));
    }
    if (rolloutIdContains) {
      checks.push(rollout.rolloutId.includes(rolloutIdContains));
    }
    if (checks.length === 0) {
      return true;
    }
    return filterLogic === 'or' ? checks.some(Boolean) : checks.every(Boolean);
  });
}

/**
 * Get the sort value for a rollout based on the sort_by field.
 */
export function getRolloutSortValue(rollout: Rollout, sortBy: string): string | number | null {
  switch (sortBy) {
    case 'rollout_id':
      return rollout.rolloutId;
    case 'status':
      return rollout.status;
    case 'mode':
      return rollout.mode ?? '';
    case 'start_time':
    default:
      return rollout.attempt?.startTime ?? rollout.startTime ?? null;
  }
}

/**
 * Sort rollouts based on query parameters.
 * Default sort_by is 'start_time', default sort_order is 'asc'.
 */
export function sortRolloutsForParams(
  rollouts: Rollout[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Rollout[] {
  const resolvedSortBy = sortBy ?? 'start_time';
  const sorted = [...rollouts].sort((a, b) => {
    const aValue = getRolloutSortValue(a, resolvedSortBy);
    const bValue = getRolloutSortValue(b, resolvedSortBy);
    if (aValue === bValue) {
      return 0;
    }
    if (aValue == null) {
      return -1;
    }
    if (bValue == null) {
      return 1;
    }
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return aValue - bValue;
    }
    return String(aValue).localeCompare(String(bValue));
  });

  if (sortOrder === 'desc') {
    sorted.reverse();
  }

  return sorted;
}

/**
 * Build a paginated rollouts response matching the Python server's format.
 * Applies filtering, sorting, and pagination based on query parameters.
 */
export function buildRolloutsResponse(rollouts: Rollout[], request: Request): Record<string, unknown> {
  const url = new URL(request.url);
  const params = url.searchParams;
  const filtered = filterRolloutsForParams(rollouts, params);
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortRolloutsForParams(filtered, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated = effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: filtered.length,
  });
}

/**
 * Sort attempts based on query parameters.
 * Default sort_by is 'sequence_id', default sort_order is 'asc'.
 */
export function sortAttemptsForParams(
  attempts: Attempt[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Attempt[] {
  const sorted = [...attempts];
  const resolvedSortBy = sortBy ?? 'sequence_id';
  sorted.sort((a, b) => {
    if (resolvedSortBy === 'start_time') {
      return a.startTime - b.startTime;
    }
    return a.sequenceId - b.sequenceId;
  });
  if (sortOrder === 'desc') {
    sorted.reverse();
  }
  return sorted;
}

/**
 * Build a paginated attempts response matching the Python server's format.
 * Applies sorting and pagination based on query parameters.
 */
export function buildAttemptsResponse(attempts: Attempt[], request: Request): Record<string, unknown> {
  const url = new URL(request.url);
  const params = url.searchParams;
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortAttemptsForParams(attempts, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated = effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: attempts.length,
  });
}

/**
 * Filter spans based on query parameters.
 * Supports: trace_id_contains, span_id_contains, name_contains
 */
export function filterSpansForParams(spans: Span[], params: URLSearchParams): Span[] {
  const traceContains = params.get('trace_id_contains');
  const spanContains = params.get('span_id_contains');
  const nameContains = params.get('name_contains');
  const filterLogic = params.get('filter_logic') === 'or' ? 'or' : 'and';

  return spans.filter((span) => {
    const checks: boolean[] = [];
    if (traceContains) {
      checks.push(span.traceId.includes(traceContains));
    }
    if (spanContains) {
      checks.push(span.spanId.includes(spanContains));
    }
    if (nameContains) {
      checks.push(span.name.toLowerCase().includes(nameContains.toLowerCase()));
    }
    if (checks.length === 0) {
      return true;
    }
    return filterLogic === 'or' ? checks.some(Boolean) : checks.every(Boolean);
  });
}

/**
 * Get the sort value for a span based on the sort_by field.
 */
export function getSpanSortValue(span: Span, sortBy: string): string | number | null {
  switch (sortBy) {
    case 'trace_id':
      return span.traceId;
    case 'span_id':
      return span.spanId;
    case 'parent_id':
      return span.parentId ?? '';
    case 'name':
      return span.name;
    case 'status_code':
      return span.status?.status_code ?? '';
    case 'duration': {
      if (span.startTime != null && span.endTime != null) {
        return span.endTime - span.startTime;
      }
      return null;
    }
    case 'start_time':
    default:
      return span.startTime ?? null;
  }
}

/**
 * Sort spans based on query parameters.
 * Default sort_by is 'start_time', default sort_order is 'asc'.
 */
export function sortSpansForParams(spans: Span[], sortBy: string | null, sortOrder: 'asc' | 'desc'): Span[] {
  const resolvedSortBy = sortBy ?? 'start_time';
  const sorted = [...spans].sort((a, b) => {
    const aValue = getSpanSortValue(a, resolvedSortBy);
    const bValue = getSpanSortValue(b, resolvedSortBy);
    if (aValue === bValue) {
      return 0;
    }
    if (aValue == null) {
      return -1;
    }
    if (bValue == null) {
      return 1;
    }
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return aValue - bValue;
    }
    return String(aValue).localeCompare(String(bValue));
  });
  if (sortOrder === 'desc') {
    sorted.reverse();
  }
  return sorted;
}

/**
 * Build a paginated spans response matching the Python server's format.
 * Applies filtering, sorting, and pagination based on query parameters.
 */
export function buildSpansResponse(spansByAttempt: Record<string, Span[]>, request: Request): Record<string, unknown> {
  const url = new URL(request.url);
  const params = url.searchParams;
  const rolloutId = params.get('rollout_id');

  if (!rolloutId) {
    return snakeCaseKeys({ items: [], limit: 0, offset: 0, total: 0 });
  }

  const attemptId = params.get('attempt_id');
  const key = attemptId ? `${rolloutId}:${attemptId}` : rolloutId;
  const spans = spansByAttempt[key] ?? [];

  const filtered = filterSpansForParams(spans, params);
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortSpansForParams(filtered, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated = effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: filtered.length,
  });
}

/**
 * Create MSW handlers for rollouts and attempts endpoints.
 *
 * @param rollouts - Array of rollout objects to serve
 * @param attemptsByRollout - Map of rollout IDs to their attempts
 * @returns Array of MSW request handlers
 *
 * @example
 * ```ts
 * const handlers = createRolloutsHandlers(sampleRollouts, attemptsByRollout);
 * ```
 */
export function createRolloutsHandlers(rollouts: Rollout[], attemptsByRollout: Record<string, Attempt[]>) {
  return [
    http.get('*/v1/agl/rollouts', ({ request }) => HttpResponse.json(buildRolloutsResponse(rollouts, request))),
    http.get('*/v1/agl/rollouts/:rolloutId/attempts', ({ params, request }) => {
      const rolloutId = params.rolloutId as string;
      const attempts = attemptsByRollout[rolloutId] ?? [];
      return HttpResponse.json(buildAttemptsResponse(attempts, request));
    }),
  ];
}

/**
 * Create MSW handlers for spans endpoints.
 *
 * @param spansByAttempt - Map of "rolloutId:attemptId" to their spans
 * @returns Array of MSW request handlers
 *
 * @example
 * ```ts
 * const handlers = createSpansHandlers({ 'ro-1:at-1': [span1, span2] });
 * ```
 */
export function createSpansHandlers(spansByAttempt: Record<string, Span[]>) {
  return [http.get('*/v1/agl/spans', ({ request }) => HttpResponse.json(buildSpansResponse(spansByAttempt, request)))];
}

/**
 * Filter resources based on query parameters.
 * Supports: resources_id_contains
 */
export function filterResourcesForParams(resources: Resources[], params: URLSearchParams): Resources[] {
  const resourcesIdContains = params.get('resources_id_contains');

  return resources.filter((resource) => {
    if (resourcesIdContains && !resource.resourcesId.includes(resourcesIdContains)) {
      return false;
    }
    return true;
  });
}

/**
 * Get the sort value for resources based on the sort_by field.
 */
export function getResourcesSortValue(resource: Resources, sortBy: string): string | number | null {
  switch (sortBy) {
    case 'resources_id':
      return resource.resourcesId;
    case 'version':
      return resource.version;
    case 'create_time':
      return resource.createTime;
    case 'update_time':
    default:
      return resource.updateTime;
  }
}

/**
 * Sort resources based on query parameters.
 * Default sort_by is 'update_time', default sort_order is 'desc'.
 */
export function sortResourcesForParams(
  resources: Resources[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Resources[] {
  const resolvedSortBy = sortBy ?? 'update_time';
  const sorted = [...resources].sort((a, b) => {
    const aValue = getResourcesSortValue(a, resolvedSortBy);
    const bValue = getResourcesSortValue(b, resolvedSortBy);
    if (aValue === bValue) {
      return 0;
    }
    if (aValue == null) {
      return -1;
    }
    if (bValue == null) {
      return 1;
    }
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return aValue - bValue;
    }
    return String(aValue).localeCompare(String(bValue));
  });

  if (sortOrder === 'desc') {
    sorted.reverse();
  }

  return sorted;
}

/**
 * Build a paginated resources response matching the Python server's format.
 * Applies filtering, sorting, and pagination based on query parameters.
 */
export function buildResourcesResponse(resources: Resources[], request: Request): Record<string, unknown> {
  const url = new URL(request.url);
  const params = url.searchParams;
  const filtered = filterResourcesForParams(resources, params);
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortResourcesForParams(filtered, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated = effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: filtered.length,
  });
}

/**
 * Filter workers based on query parameters.
 * Supports: status_in, worker_id_contains
 */
export function filterWorkersForParams(workers: Worker[], params: URLSearchParams): Worker[] {
  const statusFilters = params.getAll('status_in');
  const workerIdContains = params.get('worker_id_contains');
  const filterLogic = params.get('filter_logic') === 'or' ? 'or' : 'and';

  return workers.filter((worker) => {
    const checks: boolean[] = [];
    if (statusFilters.length > 0) {
      checks.push(statusFilters.includes(worker.status));
    }
    if (workerIdContains) {
      checks.push(worker.workerId.toLowerCase().includes(workerIdContains.toLowerCase()));
    }
    if (checks.length === 0) {
      return true;
    }
    return filterLogic === 'or' ? checks.some(Boolean) : checks.every(Boolean);
  });
}

/**
 * Resolve a worker sort value for the given column.
 */
export function getWorkerSortValue(worker: Worker, sortBy: string): string | number | null {
  switch (sortBy) {
    case 'worker_id':
      return worker.workerId;
    case 'status':
      return worker.status;
    case 'current_rollout_id':
      return worker.currentRolloutId ?? '';
    case 'current_attempt_id':
      return worker.currentAttemptId ?? '';
    case 'last_busy_time':
      return worker.lastBusyTime ?? null;
    case 'last_idle_time':
      return worker.lastIdleTime ?? null;
    case 'last_dequeue_time':
      return worker.lastDequeueTime ?? null;
    case 'last_heartbeat_time':
    default:
      return worker.lastHeartbeatTime ?? null;
  }
}

/**
 * Sort workers based on query parameters.
 * Default sort_by is 'last_heartbeat_time'.
 */
export function sortWorkersForParams(workers: Worker[], sortBy: string | null, sortOrder: 'asc' | 'desc'): Worker[] {
  const resolvedSortBy = sortBy ?? 'last_heartbeat_time';
  const sorted = [...workers].sort((a, b) => {
    const aValue = getWorkerSortValue(a, resolvedSortBy);
    const bValue = getWorkerSortValue(b, resolvedSortBy);
    if (aValue === bValue) {
      return 0;
    }
    if (aValue == null) {
      return -1;
    }
    if (bValue == null) {
      return 1;
    }
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return aValue - bValue;
    }
    return String(aValue).localeCompare(String(bValue));
  });

  if (sortOrder === 'desc') {
    sorted.reverse();
  }

  return sorted;
}

/**
 * Build a paginated workers response matching the Python server's format.
 */
export function buildWorkersResponse(workers: Worker[], request: Request): Record<string, unknown> {
  const url = new URL(request.url);
  const params = url.searchParams;
  const filtered = filterWorkersForParams(workers, params);
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortWorkersForParams(filtered, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated = effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: filtered.length,
  });
}

/**
 * Create MSW handlers for workers endpoints.
 */
export function createWorkersHandlers(workers: Worker[]) {
  return [http.get('*/v1/agl/workers', ({ request }) => HttpResponse.json(buildWorkersResponse(workers, request)))];
}

/**
 * Create MSW handlers for resources endpoints.
 *
 * @param resources - Array of resources objects to serve
 * @returns Array of MSW request handlers
 *
 * @example
 * ```ts
 * const handlers = createResourcesHandlers(sampleResources);
 * ```
 */
export function createResourcesHandlers(resources: Resources[]) {
  return [
    http.get('*/v1/agl/resources', ({ request }) => HttpResponse.json(buildResourcesResponse(resources, request))),
  ];
}

/**
 * Create complete MSW handlers for rollouts, attempts, and spans.
 *
 * This is a convenience function that combines createRolloutsHandlers and
 * createSpansHandlers for stories that need both.
 *
 * @param rollouts - Array of rollout objects to serve
 * @param attemptsByRollout - Map of rollout IDs to their attempts
 * @param spansByAttempt - Map of "rolloutId:attemptId" to their spans
 * @param delayMs - Optional delay in milliseconds to add before each response (for testing loading states)
 * @returns Array of MSW request handlers
 *
 * @example
 * ```ts
 * // Create handlers without delay
 * const handlers = createMockHandlers(rollouts, attempts, spans);
 *
 * // Create handlers with 800ms delay to test loading states
 * const loadingHandlers = createMockHandlers(rollouts, attempts, spans, 800);
 * ```
 */
export function createMockHandlers(
  rollouts: Rollout[],
  attemptsByRollout: Record<string, Attempt[]>,
  spansByAttempt?: Record<string, Span[]>,
  delayMs?: number,
) {
  // If no delay, use synchronous handlers
  if (!delayMs) {
    const handlers = createRolloutsHandlers(rollouts, attemptsByRollout);
    if (spansByAttempt) {
      handlers.push(...createSpansHandlers(spansByAttempt));
    }
    return handlers;
  }

  // Create delayed handlers
  return [
    http.get('*/v1/agl/rollouts', async ({ request }) => {
      await delay(delayMs);
      return HttpResponse.json(buildRolloutsResponse(rollouts, request));
    }),
    http.get('*/v1/agl/rollouts/:rolloutId/attempts', async ({ params, request }) => {
      await delay(delayMs);
      const rolloutId = params.rolloutId as string;
      const attempts = attemptsByRollout[rolloutId] ?? [];
      return HttpResponse.json(buildAttemptsResponse(attempts, request));
    }),
    ...(spansByAttempt
      ? [
          http.get('*/v1/agl/spans', async ({ request }) => {
            await delay(delayMs);
            return HttpResponse.json(buildSpansResponse(spansByAttempt, request));
          }),
        ]
      : []),
  ];
}
