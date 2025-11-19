// Copyright (c) Microsoft. All rights reserved.

/**
 * Tests for mock API handlers.
 *
 * These tests verify that the mock implementation behaves consistently with
 * the Python server in agentlightning/store/client_server.py.
 */

import { describe, expect, it } from 'vitest';
import type { Attempt, Resources, Rollout, Span, Worker } from '@/types';
import {
  buildAttemptsResponse,
  buildResourcesResponse,
  buildRolloutsResponse,
  buildSpansResponse,
  buildWorkersResponse,
  createMockHandlers,
  createResourcesHandlers,
  createRolloutsHandlers,
  createSpansHandlers,
  createWorkersHandlers,
  filterResourcesForParams,
  filterRolloutsForParams,
  filterSpansForParams,
  filterWorkersForParams,
  getResourcesSortValue,
  getRolloutSortValue,
  getSpanSortValue,
  getWorkerSortValue,
  parseNumberParam,
  sortAttemptsForParams,
  sortResourcesForParams,
  sortRolloutsForParams,
  sortSpansForParams,
  sortWorkersForParams,
} from './mock';

const now = Math.floor(Date.now() / 1000);

const sampleRollouts: Rollout[] = [
  {
    rolloutId: 'ro-001',
    input: { task: 'Test 1' },
    status: 'running',
    mode: 'train',
    resourcesId: 'rs-100',
    startTime: now - 1000,
    endTime: null,
    attempt: {
      rolloutId: 'ro-001',
      attemptId: 'at-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 1000,
      endTime: null,
      workerId: 'worker-1',
      lastHeartbeatTime: now - 10,
      metadata: null,
    },
    config: {},
    metadata: null,
  },
  {
    rolloutId: 'ro-002',
    input: { task: 'Test 2' },
    status: 'succeeded',
    mode: 'val',
    resourcesId: 'rs-101',
    startTime: now - 2000,
    endTime: now - 1500,
    attempt: {
      rolloutId: 'ro-002',
      attemptId: 'at-002',
      sequenceId: 1,
      status: 'succeeded',
      startTime: now - 2000,
      endTime: now - 1500,
      workerId: 'worker-2',
      lastHeartbeatTime: now - 1500,
      metadata: null,
    },
    config: {},
    metadata: null,
  },
  {
    rolloutId: 'ro-003',
    input: { task: 'Test 3' },
    status: 'failed',
    mode: 'test',
    resourcesId: null,
    startTime: now - 3000,
    endTime: now - 2500,
    attempt: {
      rolloutId: 'ro-003',
      attemptId: 'at-003',
      sequenceId: 1,
      status: 'failed',
      startTime: now - 3000,
      endTime: now - 2500,
      workerId: 'worker-3',
      lastHeartbeatTime: now - 2500,
      metadata: null,
    },
    config: {},
    metadata: null,
  },
];

const sampleAttempts: Attempt[] = [
  {
    rolloutId: 'ro-001',
    attemptId: 'at-001',
    sequenceId: 1,
    status: 'running',
    startTime: now - 1000,
    endTime: null,
    workerId: 'worker-1',
    lastHeartbeatTime: now - 10,
    metadata: null,
  },
  {
    rolloutId: 'ro-001',
    attemptId: 'at-002',
    sequenceId: 2,
    status: 'failed',
    startTime: now - 900,
    endTime: now - 800,
    workerId: 'worker-1',
    lastHeartbeatTime: now - 800,
    metadata: null,
  },
  {
    rolloutId: 'ro-001',
    attemptId: 'at-003',
    sequenceId: 3,
    status: 'running',
    startTime: now - 700,
    endTime: null,
    workerId: 'worker-2',
    lastHeartbeatTime: now - 5,
    metadata: null,
  },
];

const sampleSpans: Span[] = [
  {
    rolloutId: 'ro-001',
    attemptId: 'at-001',
    sequenceId: 1,
    traceId: 'tr-001',
    spanId: 'sp-001',
    parentId: null,
    name: 'Initialize',
    status: { status_code: 'OK', description: null },
    attributes: {},
    startTime: now - 1000,
    endTime: now - 950,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-001',
    attemptId: 'at-001',
    sequenceId: 2,
    traceId: 'tr-001',
    spanId: 'sp-002',
    parentId: 'sp-001',
    name: 'Process',
    status: { status_code: 'OK', description: null },
    attributes: {},
    startTime: now - 950,
    endTime: now - 900,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-001',
    attemptId: 'at-001',
    sequenceId: 3,
    traceId: 'tr-002',
    spanId: 'sp-003',
    parentId: null,
    name: 'Finalize',
    status: { status_code: 'ERROR', description: 'Failed' },
    attributes: {},
    startTime: now - 850,
    endTime: now - 800,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
];

const sampleResources: Resources[] = [
  {
    resourcesId: 'rs-001',
    version: 2,
    createTime: now - 400,
    updateTime: now - 200,
    resources: { config: { learning_rate: 0.01 } },
  },
  {
    resourcesId: 'rs-002',
    version: 3,
    createTime: now - 300,
    updateTime: now - 100,
    resources: { config: { learning_rate: 0.001 } },
  },
  {
    resourcesId: 'rs-003',
    version: 1,
    createTime: now - 200,
    updateTime: now - 50,
    resources: { config: { learning_rate: 0.1 } },
  },
];

const sampleWorkers: Worker[] = [
  {
    workerId: 'worker-alpha',
    status: 'busy',
    heartbeatStats: { queueDepth: 2 },
    lastHeartbeatTime: now - 30,
    lastDequeueTime: now - 300,
    lastBusyTime: now - 60,
    lastIdleTime: now - 600,
    currentRolloutId: 'ro-001',
    currentAttemptId: 'at-001',
  },
  {
    workerId: 'worker-beta',
    status: 'idle',
    heartbeatStats: { queueDepth: 0 },
    lastHeartbeatTime: now - 120,
    lastDequeueTime: now - 1200,
    lastBusyTime: now - 3600,
    lastIdleTime: now - 180,
    currentRolloutId: null,
    currentAttemptId: null,
  },
  {
    workerId: 'worker-gamma',
    status: 'busy',
    heartbeatStats: null,
    lastHeartbeatTime: now - 10,
    lastDequeueTime: now - 60,
    lastBusyTime: now - 20,
    lastIdleTime: now - 4000,
    currentRolloutId: 'ro-003',
    currentAttemptId: 'at-003',
  },
  {
    workerId: 'worker-delta',
    status: 'unknown',
    heartbeatStats: { queueDepth: 0 },
    lastHeartbeatTime: now - 5,
    lastDequeueTime: now - 80,
    lastBusyTime: null,
    lastIdleTime: null,
    currentRolloutId: null,
    currentAttemptId: null,
  },
];

describe('parseNumberParam', () => {
  it('returns default value when param is missing', () => {
    const params = new URLSearchParams();
    expect(parseNumberParam(params, 'limit', 10)).toBe(10);
  });

  it('parses valid number', () => {
    const params = new URLSearchParams('limit=25');
    expect(parseNumberParam(params, 'limit', 10)).toBe(25);
  });

  it('returns default for invalid number', () => {
    const params = new URLSearchParams('limit=invalid');
    expect(parseNumberParam(params, 'limit', 10)).toBe(10);
  });

  it('returns default for NaN', () => {
    const params = new URLSearchParams('limit=NaN');
    expect(parseNumberParam(params, 'limit', 10)).toBe(10);
  });

  it('handles negative numbers', () => {
    const params = new URLSearchParams('offset=-1');
    expect(parseNumberParam(params, 'offset', 0)).toBe(-1);
  });
});

describe('filterRolloutsForParams', () => {
  it('returns all rollouts when no filters are provided', () => {
    const params = new URLSearchParams();
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result).toHaveLength(3);
  });

  it('filters by single status', () => {
    const params = new URLSearchParams('status_in=running');
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result).toHaveLength(1);
    expect(result[0].status).toBe('running');
  });

  it('filters by multiple statuses', () => {
    const params = new URLSearchParams();
    params.append('status_in', 'running');
    params.append('status_in', 'succeeded');
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result).toHaveLength(2);
  });

  it('filters by mode', () => {
    const params = new URLSearchParams();
    params.append('mode_in', 'train');
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result).toHaveLength(1);
    expect(result[0].mode).toBe('train');
  });

  it('filters by rollout_id_contains', () => {
    const params = new URLSearchParams('rollout_id_contains=002');
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result).toHaveLength(1);
    expect(result[0].rolloutId).toBe('ro-002');
  });

  it('applies multiple filters together', () => {
    const params = new URLSearchParams();
    params.append('status_in', 'running');
    params.append('mode_in', 'train');
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result).toHaveLength(1);
    expect(result[0].rolloutId).toBe('ro-001');
  });

  it('supports filter_logic=or across filters', () => {
    const params = new URLSearchParams();
    params.append('status_in', 'failed');
    params.append('mode_in', 'train');
    params.set('filter_logic', 'or');
    const result = filterRolloutsForParams(sampleRollouts, params);
    expect(result.map((r) => r.rolloutId).sort()).toEqual(['ro-001', 'ro-003']);
  });
});

describe('getRolloutSortValue', () => {
  const rollout = sampleRollouts[0];

  it('returns rollout_id for rollout_id sort', () => {
    expect(getRolloutSortValue(rollout, 'rollout_id')).toBe('ro-001');
  });

  it('returns status for status sort', () => {
    expect(getRolloutSortValue(rollout, 'status')).toBe('running');
  });

  it('returns mode for mode sort', () => {
    expect(getRolloutSortValue(rollout, 'mode')).toBe('train');
  });

  it('returns empty string for null mode', () => {
    const rolloutWithoutMode = { ...rollout, mode: null };
    expect(getRolloutSortValue(rolloutWithoutMode, 'mode')).toBe('');
  });

  it('returns start_time as default', () => {
    expect(getRolloutSortValue(rollout, 'unknown_field')).toBe(now - 1000);
  });

  it('falls back to rollout.startTime when attempt is null', () => {
    const rolloutWithoutAttempt = { ...rollout, attempt: null };
    expect(getRolloutSortValue(rolloutWithoutAttempt, 'start_time')).toBe(now - 1000);
  });
});

describe('sortRolloutsForParams', () => {
  it('sorts by start_time ascending by default', () => {
    const result = sortRolloutsForParams(sampleRollouts, null, 'asc');
    expect(result[0].rolloutId).toBe('ro-003');
    expect(result[2].rolloutId).toBe('ro-001');
  });

  it('sorts by start_time descending', () => {
    const result = sortRolloutsForParams(sampleRollouts, 'start_time', 'desc');
    expect(result[0].rolloutId).toBe('ro-001');
    expect(result[2].rolloutId).toBe('ro-003');
  });

  it('sorts by rollout_id ascending', () => {
    const result = sortRolloutsForParams(sampleRollouts, 'rollout_id', 'asc');
    expect(result[0].rolloutId).toBe('ro-001');
    expect(result[2].rolloutId).toBe('ro-003');
  });

  it('sorts by status', () => {
    const result = sortRolloutsForParams(sampleRollouts, 'status', 'asc');
    expect(result[0].status).toBe('failed');
    expect(result[1].status).toBe('running');
    expect(result[2].status).toBe('succeeded');
  });

  it('handles null values in sorting', () => {
    const rolloutsWithNull: Rollout[] = [{ ...sampleRollouts[0], attempt: null, startTime: 0 }, sampleRollouts[1]];
    const result = sortRolloutsForParams(rolloutsWithNull, 'start_time', 'asc');
    expect(result[0].attempt).toBeNull();
  });
});

describe('buildRolloutsResponse', () => {
  it('returns all rollouts with default pagination', () => {
    const request = new Request('http://localhost/v1/agl/rollouts');
    const response = buildRolloutsResponse(sampleRollouts, request);
    expect(response.items).toHaveLength(3);
    expect(response.total).toBe(3);
    expect(response.limit).toBe(3);
    expect(response.offset).toBe(0);
  });

  it('applies limit pagination', () => {
    const request = new Request('http://localhost/v1/agl/rollouts?limit=2');
    const response = buildRolloutsResponse(sampleRollouts, request);
    expect(response.items).toHaveLength(2);
    expect(response.limit).toBe(2);
    expect(response.total).toBe(3);
  });

  it('applies offset pagination', () => {
    const request = new Request('http://localhost/v1/agl/rollouts?offset=1');
    const response = buildRolloutsResponse(sampleRollouts, request);
    expect(response.items).toHaveLength(2);
    expect(response.offset).toBe(1);
  });

  it('applies limit and offset together', () => {
    const request = new Request('http://localhost/v1/agl/rollouts?limit=1&offset=1');
    const response = buildRolloutsResponse(sampleRollouts, request);
    expect(response.items).toHaveLength(1);
    expect(response.limit).toBe(1);
    expect(response.offset).toBe(1);
  });

  it('handles negative limit (returns all)', () => {
    const request = new Request('http://localhost/v1/agl/rollouts?limit=-1');
    const response = buildRolloutsResponse(sampleRollouts, request);
    expect(response.items).toHaveLength(3);
    expect(response.limit).toBe(3);
  });

  it('applies filtering before pagination', () => {
    const request = new Request('http://localhost/v1/agl/rollouts?status_in=running&limit=10');
    const response = buildRolloutsResponse(sampleRollouts, request);
    expect(response.items).toHaveLength(1);
    expect(response.total).toBe(1);
  });

  it('applies sorting before pagination', () => {
    const request = new Request('http://localhost/v1/agl/rollouts?sort_by=rollout_id&sort_order=desc&limit=1');
    const response = buildRolloutsResponse(sampleRollouts, request);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0].rollout_id).toBe('ro-003');
  });

  it('uses snake_case keys in response', () => {
    const request = new Request('http://localhost/v1/agl/rollouts');
    const response = buildRolloutsResponse(sampleRollouts, request);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0]).toHaveProperty('rollout_id');
    expect(items[0]).toHaveProperty('start_time');
    expect(items[0]).not.toHaveProperty('rolloutId');
  });
});

describe('sortAttemptsForParams', () => {
  it('sorts by sequence_id ascending by default', () => {
    const result = sortAttemptsForParams(sampleAttempts, null, 'asc');
    expect(result[0].sequenceId).toBe(1);
    expect(result[2].sequenceId).toBe(3);
  });

  it('sorts by sequence_id descending', () => {
    const result = sortAttemptsForParams(sampleAttempts, 'sequence_id', 'desc');
    expect(result[0].sequenceId).toBe(3);
    expect(result[2].sequenceId).toBe(1);
  });

  it('sorts by start_time', () => {
    const result = sortAttemptsForParams(sampleAttempts, 'start_time', 'asc');
    expect(result[0].startTime).toBe(now - 1000);
    expect(result[2].startTime).toBe(now - 700);
  });
});

describe('buildAttemptsResponse', () => {
  it('returns all attempts with default pagination', () => {
    const request = new Request('http://localhost/v1/agl/rollouts/ro-001/attempts');
    const response = buildAttemptsResponse(sampleAttempts, request);
    expect(response.items).toHaveLength(3);
    expect(response.total).toBe(3);
  });

  it('applies pagination correctly', () => {
    const request = new Request('http://localhost/v1/agl/rollouts/ro-001/attempts?limit=2&offset=1');
    const response = buildAttemptsResponse(sampleAttempts, request);
    expect(response.items).toHaveLength(2);
    expect(response.offset).toBe(1);
  });

  it('sorts before paginating', () => {
    const request = new Request(
      'http://localhost/v1/agl/rollouts/ro-001/attempts?sort_by=sequence_id&sort_order=desc&limit=1',
    );
    const response = buildAttemptsResponse(sampleAttempts, request);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0].sequence_id).toBe(3);
  });
});

describe('filterSpansForParams', () => {
  it('returns all spans when no filters are provided', () => {
    const params = new URLSearchParams();
    const result = filterSpansForParams(sampleSpans, params);
    expect(result).toHaveLength(3);
  });

  it('filters by trace_id_contains', () => {
    const params = new URLSearchParams('trace_id_contains=tr-001');
    const result = filterSpansForParams(sampleSpans, params);
    expect(result).toHaveLength(2);
  });

  it('filters by span_id_contains', () => {
    const params = new URLSearchParams('span_id_contains=sp-003');
    const result = filterSpansForParams(sampleSpans, params);
    expect(result).toHaveLength(1);
    expect(result[0].spanId).toBe('sp-003');
  });

  it('filters by name_contains (case insensitive)', () => {
    const params = new URLSearchParams('name_contains=INIT');
    const result = filterSpansForParams(sampleSpans, params);
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('Initialize');
  });

  it('applies multiple filters together', () => {
    const params = new URLSearchParams();
    params.set('trace_id_contains', 'tr-001');
    params.set('name_contains', 'Process');
    const result = filterSpansForParams(sampleSpans, params);
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('Process');
  });

  it('supports filter_logic=or for spans', () => {
    const params = new URLSearchParams();
    params.set('trace_id_contains', 'tr-001');
    params.set('name_contains', 'Finalize');
    params.set('filter_logic', 'or');
    const result = filterSpansForParams(sampleSpans, params);
    expect(result).toHaveLength(3);
  });
});

describe('getSpanSortValue', () => {
  const span = sampleSpans[0];

  it('returns trace_id for trace_id sort', () => {
    expect(getSpanSortValue(span, 'trace_id')).toBe('tr-001');
  });

  it('returns span_id for span_id sort', () => {
    expect(getSpanSortValue(span, 'span_id')).toBe('sp-001');
  });

  it('returns parent_id for parent_id sort', () => {
    expect(getSpanSortValue(sampleSpans[1], 'parent_id')).toBe('sp-001');
  });

  it('returns empty string for null parent_id', () => {
    expect(getSpanSortValue(span, 'parent_id')).toBe('');
  });

  it('returns name for name sort', () => {
    expect(getSpanSortValue(span, 'name')).toBe('Initialize');
  });

  it('returns status_code for status_code sort', () => {
    expect(getSpanSortValue(span, 'status_code')).toBe('OK');
  });

  it('calculates duration', () => {
    const duration = getSpanSortValue(span, 'duration');
    expect(duration).toBe(50);
  });

  it('returns null duration when times are missing', () => {
    const spanWithoutEnd: Span = { ...span, endTime: now - 950 };
    expect(getSpanSortValue(spanWithoutEnd, 'duration')).toBe(50);
  });

  it('returns start_time as default', () => {
    expect(getSpanSortValue(span, 'unknown_field')).toBe(now - 1000);
  });
});

describe('sortSpansForParams', () => {
  it('sorts by start_time ascending by default', () => {
    const result = sortSpansForParams(sampleSpans, null, 'asc');
    expect(result[0].spanId).toBe('sp-001');
    expect(result[2].spanId).toBe('sp-003');
  });

  it('sorts by name', () => {
    const result = sortSpansForParams(sampleSpans, 'name', 'asc');
    expect(result[0].name).toBe('Finalize');
    expect(result[2].name).toBe('Process');
  });

  it('sorts by duration', () => {
    const result = sortSpansForParams(sampleSpans, 'duration', 'desc');
    // All spans have duration of 50, so we just check that sorting works
    expect(result[0].endTime! - result[0].startTime!).toBe(50);
  });
});

describe('buildSpansResponse', () => {
  const spansByAttempt = {
    'ro-001:at-001': sampleSpans,
  };

  it('returns empty when rollout_id is missing', () => {
    const request = new Request('http://localhost/v1/agl/spans');
    const response = buildSpansResponse(spansByAttempt, request);
    expect(response.items).toHaveLength(0);
    expect(response.total).toBe(0);
  });

  it('returns spans for rollout and attempt', () => {
    const request = new Request('http://localhost/v1/agl/spans?rollout_id=ro-001&attempt_id=at-001');
    const response = buildSpansResponse(spansByAttempt, request);
    expect(response.items).toHaveLength(3);
    expect(response.total).toBe(3);
  });

  it('handles missing attempt_id', () => {
    const request = new Request('http://localhost/v1/agl/spans?rollout_id=ro-001');
    const response = buildSpansResponse(spansByAttempt, request);
    expect(response.items).toHaveLength(0);
  });

  it('applies filtering', () => {
    const request = new Request(
      'http://localhost/v1/agl/spans?rollout_id=ro-001&attempt_id=at-001&trace_id_contains=tr-001',
    );
    const response = buildSpansResponse(spansByAttempt, request);
    expect(response.items).toHaveLength(2);
  });

  it('applies sorting and pagination', () => {
    const request = new Request(
      'http://localhost/v1/agl/spans?rollout_id=ro-001&attempt_id=at-001&sort_by=name&limit=2',
    );
    const response = buildSpansResponse(spansByAttempt, request);
    expect(response.items).toHaveLength(2);
    const items = response.items as Span[];
    expect(items[0].name).toBe('Finalize');
  });

  it('supports filter_logic=or in responses', () => {
    const request = new Request(
      'http://localhost/v1/agl/spans?rollout_id=ro-001&attempt_id=at-001&trace_id_contains=tr-001&name_contains=Finalize&filter_logic=or',
    );
    const response = buildSpansResponse(spansByAttempt, request);
    expect(response.items).toHaveLength(3);
  });
});

describe('filterResourcesForParams', () => {
  it('returns all resources when no filter is applied', () => {
    const params = new URLSearchParams();
    const result = filterResourcesForParams(sampleResources, params);
    expect(result).toHaveLength(3);
  });

  it('filters resources by resources_id_contains', () => {
    const params = new URLSearchParams('resources_id_contains=002');
    const result = filterResourcesForParams(sampleResources, params);
    expect(result).toHaveLength(1);
    expect(result[0].resourcesId).toBe('rs-002');
  });
});

describe('getResourcesSortValue', () => {
  const resource = sampleResources[0];

  it('returns resources_id value', () => {
    expect(getResourcesSortValue(resource, 'resources_id')).toBe('rs-001');
  });

  it('returns version value', () => {
    expect(getResourcesSortValue(resource, 'version')).toBe(2);
  });

  it('returns create_time value', () => {
    expect(getResourcesSortValue(resource, 'create_time')).toBe(now - 400);
  });

  it('returns update_time for unknown sort keys', () => {
    expect(getResourcesSortValue(resource, 'unknown')).toBe(now - 200);
  });
});

describe('sortResourcesForParams', () => {
  it('sorts by update_time ascending by default', () => {
    const result = sortResourcesForParams(sampleResources, null, 'asc');
    expect(result[0].resourcesId).toBe('rs-001');
    expect(result[2].resourcesId).toBe('rs-003');
  });

  it('sorts by update_time descending', () => {
    const result = sortResourcesForParams(sampleResources, 'update_time', 'desc');
    expect(result[0].resourcesId).toBe('rs-003');
    expect(result[2].resourcesId).toBe('rs-001');
  });

  it('sorts by version', () => {
    const result = sortResourcesForParams(sampleResources, 'version', 'asc');
    expect(result[0].version).toBe(1);
    expect(result[2].version).toBe(3);
  });
});

describe('buildResourcesResponse', () => {
  it('returns paginated resources with defaults', () => {
    const request = new Request('http://localhost/v1/agl/resources');
    const response = buildResourcesResponse(sampleResources, request);
    expect(response.items).toHaveLength(3);
    expect(response.offset).toBe(0);
    expect(response.limit).toBe(3);
    expect(response.total).toBe(3);
  });

  it('applies filter before pagination', () => {
    const request = new Request('http://localhost/v1/agl/resources?resources_id_contains=003');
    const response = buildResourcesResponse(sampleResources, request);
    expect(response.items).toHaveLength(1);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0].resources_id).toBe('rs-003');
  });

  it('applies sort order and pagination parameters', () => {
    const request = new Request('http://localhost/v1/agl/resources?sort_by=version&sort_order=desc&limit=1&offset=1');
    const response = buildResourcesResponse(sampleResources, request);
    expect(response.items).toHaveLength(1);
    expect(response.offset).toBe(1);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0].version).toBe(2);
  });
});

describe('createResourcesHandlers', () => {
  it('creates handler for resources endpoint', () => {
    const handlers = createResourcesHandlers(sampleResources);
    expect(handlers).toHaveLength(1);
    expect(handlers[0].info.header).toContain('GET');
  });
});

describe('filterWorkersForParams', () => {
  it('returns all workers without filters', () => {
    const params = new URLSearchParams();
    const result = filterWorkersForParams(sampleWorkers, params);
    expect(result).toHaveLength(4);
  });

  it('filters by status and worker ID substring using AND logic', () => {
    const params = new URLSearchParams('status_in=busy&worker_id_contains=gamma');
    const result = filterWorkersForParams(sampleWorkers, params);
    expect(result).toHaveLength(1);
    expect(result[0].workerId).toBe('worker-gamma');
  });

  it('supports filter_logic=or', () => {
    const params = new URLSearchParams('status_in=idle&worker_id_contains=gamma&filter_logic=or');
    const result = filterWorkersForParams(sampleWorkers, params);
    expect(result).toHaveLength(2);
  });

  it('filters by unknown status', () => {
    const params = new URLSearchParams('status_in=unknown');
    const result = filterWorkersForParams(sampleWorkers, params);
    expect(result).toHaveLength(1);
    expect(result[0].workerId).toBe('worker-delta');
  });
});

describe('getWorkerSortValue', () => {
  const worker = sampleWorkers[0];

  it('returns worker_id', () => {
    expect(getWorkerSortValue(worker, 'worker_id')).toBe('worker-alpha');
  });

  it('returns status', () => {
    expect(getWorkerSortValue(worker, 'status')).toBe('busy');
  });

  it('returns timestamp fields', () => {
    expect(getWorkerSortValue(worker, 'last_busy_time')).toBe(worker.lastBusyTime);
    expect(getWorkerSortValue(worker, 'last_idle_time')).toBe(worker.lastIdleTime);
    expect(getWorkerSortValue(worker, 'last_dequeue_time')).toBe(worker.lastDequeueTime);
  });

  it('returns rollout and attempt identifiers', () => {
    expect(getWorkerSortValue(worker, 'current_rollout_id')).toBe(worker.currentRolloutId);
    expect(getWorkerSortValue(worker, 'current_attempt_id')).toBe(worker.currentAttemptId);
  });

  it('falls back to last_heartbeat_time', () => {
    expect(getWorkerSortValue(worker, 'unknown')).toBe(worker.lastHeartbeatTime);
  });
});

describe('sortWorkersForParams', () => {
  it('sorts by last heartbeat ascending by default', () => {
    const result = sortWorkersForParams(sampleWorkers, null, 'asc');
    expect(result.map((worker) => worker.workerId)).toEqual([
      'worker-beta',
      'worker-alpha',
      'worker-gamma',
      'worker-delta',
    ]);
  });

  it('sorts descending by worker_id when requested', () => {
    const result = sortWorkersForParams(sampleWorkers, 'worker_id', 'desc');
    expect(result.map((worker) => worker.workerId)).toEqual([
      'worker-gamma',
      'worker-delta',
      'worker-beta',
      'worker-alpha',
    ]);
  });

  it('sorts by current_rollout_id', () => {
    const result = sortWorkersForParams(sampleWorkers, 'current_rollout_id', 'asc');
    expect(result.map((worker) => worker.currentRolloutId)).toEqual([null, null, 'ro-001', 'ro-003']);
  });
});

describe('buildWorkersResponse', () => {
  it('applies filters before pagination', () => {
    const request = new Request('http://localhost/v1/agl/workers?worker_id_contains=beta&limit=5');
    const response = buildWorkersResponse(sampleWorkers, request);
    expect(response.items).toHaveLength(1);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0].worker_id).toBe('worker-beta');
  });

  it('applies sort and pagination parameters', () => {
    const request = new Request('http://localhost/v1/agl/workers?sort_by=worker_id&limit=2&offset=1');
    const response = buildWorkersResponse(sampleWorkers, request);
    expect(response.items).toHaveLength(2);
    const items = response.items as Array<Record<string, unknown>>;
    expect(items[0].worker_id).toBe('worker-beta');
    expect(response.total).toBe(4);
  });
});

describe('createWorkersHandlers', () => {
  it('creates handler for workers endpoint', () => {
    const handlers = createWorkersHandlers(sampleWorkers);
    expect(handlers).toHaveLength(1);
    expect(handlers[0].info.header).toContain('GET');
  });
});

describe('createRolloutsHandlers', () => {
  it('creates handlers that return correct rollout data', async () => {
    const attemptsByRollout = { 'ro-001': sampleAttempts };
    const handlers = createRolloutsHandlers(sampleRollouts, attemptsByRollout);

    expect(handlers).toHaveLength(2);
    expect(handlers[0].info.header).toContain('GET');
    expect(handlers[1].info.header).toContain('GET');
  });
});

describe('createSpansHandlers', () => {
  it('creates handler for spans endpoint', () => {
    const spansByAttempt = { 'ro-001:at-001': sampleSpans };
    const handlers = createSpansHandlers(spansByAttempt);

    expect(handlers).toHaveLength(1);
    expect(handlers[0].info.header).toContain('GET');
  });
});

describe('createMockHandlers', () => {
  it('creates combined handlers for rollouts and attempts', () => {
    const attemptsByRollout = { 'ro-001': sampleAttempts };
    const handlers = createMockHandlers(sampleRollouts, attemptsByRollout);

    expect(handlers).toHaveLength(2);
  });

  it('includes spans handlers when provided', () => {
    const attemptsByRollout = { 'ro-001': sampleAttempts };
    const spansByAttempt = { 'ro-001:at-001': sampleSpans };
    const handlers = createMockHandlers(sampleRollouts, attemptsByRollout, spansByAttempt);

    expect(handlers).toHaveLength(3);
  });

  it('works without spans handlers', () => {
    const attemptsByRollout = { 'ro-001': sampleAttempts };
    const handlers = createMockHandlers(sampleRollouts, attemptsByRollout);

    expect(handlers).toHaveLength(2);
  });
});
