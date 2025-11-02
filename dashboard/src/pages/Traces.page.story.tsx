import type { Meta, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { http, HttpResponse, delay } from 'msw';
import { TracesPage } from './Traces.page';
import { AppDrawer } from '@/components/AppDrawer.component';
import { createAppStore } from '../store';
import { initialConfigState } from '../features/config/slice';
import { initialRolloutsUiState } from '../features/rollouts/slice';
import { initialResourcesUiState } from '../features/resources/slice';
import { initialTracesUiState, type TracesUiState } from '../features/traces/slice';
import type { Attempt, Rollout, Span } from '../types';
import { snakeCaseKeys } from '@/utils/format';

const meta: Meta<typeof TracesPage> = {
  title: 'Pages/TracesPage',
  component: TracesPage,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof TracesPage>;

const now = Math.floor(Date.now() / 1000);

const sampleRollouts: Rollout[] = [
  {
    rolloutId: 'ro-traces-001',
    input: { task: 'Generate onboarding flow' },
    status: 'running',
    mode: 'train',
    resourcesId: 'rs-traces-001',
    startTime: now - 1800,
    endTime: null,
    attempt: {
      rolloutId: 'ro-traces-001',
      attemptId: 'at-traces-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 1800,
      endTime: null,
      workerId: 'worker-delta',
      lastHeartbeatTime: now - 30,
      metadata: { region: 'us-east-1' },
    },
    config: { retries: 0 },
    metadata: { owner: 'ava' },
  },
  {
    rolloutId: 'ro-traces-002',
    input: { task: 'Classify support emails' },
    status: 'succeeded',
    mode: 'val',
    resourcesId: 'rs-traces-002',
    startTime: now - 5400,
    endTime: now - 3600,
    attempt: {
      rolloutId: 'ro-traces-002',
      attemptId: 'at-traces-004',
      sequenceId: 4,
      status: 'succeeded',
      startTime: now - 4000,
      endTime: now - 3600,
      workerId: 'worker-epsilon',
      lastHeartbeatTime: now - 3600,
      metadata: { region: 'us-west-2' },
    },
    config: { retries: 2 },
    metadata: { owner: 'ben' },
  },
];

const attemptsByRollout: Record<string, Attempt[]> = {
  'ro-traces-001': [
    {
      rolloutId: 'ro-traces-001',
      attemptId: 'at-traces-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 1800,
      endTime: null,
      workerId: 'worker-delta',
      lastHeartbeatTime: now - 30,
      metadata: { region: 'us-east-1' },
    },
    {
      rolloutId: 'ro-traces-001',
      attemptId: 'at-traces-002',
      sequenceId: 2,
      status: 'failed',
      startTime: now - 5400,
      endTime: now - 4800,
      workerId: 'worker-theta',
      lastHeartbeatTime: now - 4800,
      metadata: { error: 'Network timeout' },
    },
  ],
  'ro-traces-002': [
    {
      rolloutId: 'ro-traces-002',
      attemptId: 'at-traces-004',
      sequenceId: 4,
      status: 'succeeded',
      startTime: now - 4000,
      endTime: now - 3600,
      workerId: 'worker-epsilon',
      lastHeartbeatTime: now - 3600,
      metadata: { region: 'us-west-2' },
    },
  ],
};

const spansByAttempt: Record<string, Span[]> = {
  'ro-traces-001:at-traces-001': [
    {
      rolloutId: 'ro-traces-001',
      attemptId: 'at-traces-001',
      sequenceId: 1,
      traceId: 'tr-001',
      spanId: 'sp-001',
      parentId: null,
      name: 'Initialize rollout',
      status: { status_code: 'OK', description: null },
      attributes: { stage: 'init', duration_ms: 120 },
      startTime: now - 1600,
      endTime: now - 1580,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
    {
      rolloutId: 'ro-traces-001',
      attemptId: 'at-traces-001',
      sequenceId: 2,
      traceId: 'tr-001',
      spanId: 'sp-002',
      parentId: 'sp-001',
      name: 'Fetch resources',
      status: { status_code: 'OK', description: null },
      attributes: { endpoint: '/resources/latest', duration_ms: 240 },
      startTime: now - 1580,
      endTime: now - 1540,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ],
  'ro-traces-001:at-traces-002': [
    {
      rolloutId: 'ro-traces-001',
      attemptId: 'at-traces-002',
      sequenceId: 1,
      traceId: 'tr-002',
      spanId: 'sp-101',
      parentId: null,
      name: 'Initialize rollout',
      status: { status_code: 'ERROR', description: 'Timeout' },
      attributes: { stage: 'init', duration_ms: 600 },
      startTime: now - 5300,
      endTime: now - 4700,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ],
  'ro-traces-002:at-traces-004': [
    {
      rolloutId: 'ro-traces-002',
      attemptId: 'at-traces-004',
      sequenceId: 1,
      traceId: 'tr-200',
      spanId: 'sp-201',
      parentId: null,
      name: 'Load dataset',
      status: { status_code: 'OK', description: null },
      attributes: { records: 1200, duration_ms: 420 },
      startTime: now - 3800,
      endTime: now - 3720,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
    {
      rolloutId: 'ro-traces-002',
      attemptId: 'at-traces-004',
      sequenceId: 2,
      traceId: 'tr-200',
      spanId: 'sp-202',
      parentId: 'sp-201',
      name: 'Classify batch',
      status: { status_code: 'OK', description: null },
      attributes: { batch: 1, duration_ms: 320 },
      startTime: now - 3720,
      endTime: now - 3660,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ],
};

const parseNumberParam = (params: URLSearchParams, key: string, defaultValue: number): number => {
  const raw = params.get(key);
  if (raw == null) {
    return defaultValue;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    return defaultValue;
  }
  return value;
};

const filterRolloutsForParams = (rollouts: Rollout[], params: URLSearchParams): Rollout[] => {
  const statusFilters = params.getAll('status_in');
  const modeFilters = params.getAll('mode_in');
  const rolloutIdContains = params.get('rollout_id_contains');

  return rollouts.filter((rollout) => {
    if (statusFilters.length > 0 && !statusFilters.includes(rollout.status)) {
      return false;
    }
    if (modeFilters.length > 0 && (!rollout.mode || !modeFilters.includes(rollout.mode))) {
      return false;
    }
    if (rolloutIdContains && !rollout.rolloutId.includes(rolloutIdContains)) {
      return false;
    }
    return true;
  });
};

const getRolloutSortValue = (rollout: Rollout, sortBy: string): string | number | null => {
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
};

const sortRolloutsForParams = (
  rollouts: Rollout[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Rollout[] => {
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
};

const buildRolloutsResponse = (request: Request) => {
  const url = new URL(request.url);
  const params = url.searchParams;
  const filtered = filterRolloutsForParams(sampleRollouts, params);
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortRolloutsForParams(filtered, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated =
    effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: filtered.length,
  });
};

const sortAttemptsForParams = (
  attempts: Attempt[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Attempt[] => {
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
};

const buildAttemptsResponse = (rolloutId: string, request: Request) => {
  const attempts = attemptsByRollout[rolloutId] ?? [];
  const url = new URL(request.url);
  const params = url.searchParams;
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortAttemptsForParams(attempts, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated =
    effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: attempts.length,
  });
};

const filterSpansForParams = (spans: Span[], params: URLSearchParams): Span[] => {
  const traceContains = params.get('trace_id_contains');
  const spanContains = params.get('span_id_contains');
  const nameContains = params.get('name_contains');

  return spans.filter((span) => {
    if (traceContains && !span.traceId.includes(traceContains)) {
      return false;
    }
    if (spanContains && !span.spanId.includes(spanContains)) {
      return false;
    }
    if (nameContains && !span.name.toLowerCase().includes(nameContains.toLowerCase())) {
      return false;
    }
    return true;
  });
};

const getSpanSortValue = (span: Span, sortBy: string): string | number | null => {
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
};

const sortSpansForParams = (
  spans: Span[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Span[] => {
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
};

const buildSpansResponse = (request: Request) => {
  const url = new URL(request.url);
  const params = url.searchParams;
  const rolloutId = params.get('rollout_id');
  if (!rolloutId) {
    return snakeCaseKeys({ items: [], limit: 0, offset: 0, total: 0 });
  }
  const attemptId = params.get('attempt_id');
  const spans = getSpans(rolloutId, attemptId);
  const filtered = filterSpansForParams(spans, params);
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortSpansForParams(filtered, sortBy, sortOrder);
  const limitParam = parseNumberParam(params, 'limit', sorted.length);
  const offsetParam = parseNumberParam(params, 'offset', 0);
  const effectiveLimit = limitParam < 0 ? sorted.length : limitParam;
  const offset = offsetParam < 0 ? 0 : offsetParam;
  const paginated =
    effectiveLimit >= 0 ? sorted.slice(offset, offset + effectiveLimit) : [...sorted];

  return snakeCaseKeys({
    items: paginated,
    limit: effectiveLimit,
    offset,
    total: filtered.length,
  });
};

function getSpans(rolloutId: string, attemptId?: string | null) {
  const key = `${rolloutId}:${attemptId ?? 'latest'}`;
  if (attemptId) {
    return spansByAttempt[`${rolloutId}:${attemptId}`] ?? [];
  }

  const latestAttempt = attemptsByRollout[rolloutId]?.at(-1);
  if (!latestAttempt) {
    return [];
  }
  return spansByAttempt[`${rolloutId}:${latestAttempt.attemptId}`] ?? [];
}

function createHandlers(delayMs = 0) {
  return [
    http.get('*/agl/v1/rollouts', async ({ request }) => {
      if (delayMs) {
        await delay(delayMs);
      }
      return HttpResponse.json(buildRolloutsResponse(request));
    }),
    http.get('*/agl/v1/rollouts/:rolloutId/attempts', async ({ params, request }) => {
      const rolloutId = params.rolloutId as string;
      if (delayMs) {
        await delay(delayMs);
      }
      return HttpResponse.json(buildAttemptsResponse(rolloutId, request));
    }),
    http.get('*/agl/v1/spans', async ({ request }) => {
      if (delayMs) {
        await delay(delayMs);
      }
      return HttpResponse.json(buildSpansResponse(request));
    }),
  ];
}

function renderTracesPage(preloadedTracesState?: Partial<TracesUiState>) {
  const store = createAppStore({
    config: initialConfigState,
    rollouts: initialRolloutsUiState,
    resources: initialResourcesUiState,
    traces: { ...initialTracesUiState, ...preloadedTracesState },
  });

  return (
    <Provider store={store}>
      <TracesPage />
      <AppDrawer />
    </Provider>
  );
}

export const DefaultView: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createHandlers(),
    },
  },
};

export const LoadingState: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createHandlers(800),
    },
  },
};

export const AttemptScoped: Story = {
  render: () =>
    renderTracesPage({
      attemptId: 'at-traces-002',
      rolloutId: 'ro-traces-001',
    }),
  parameters: {
    msw: {
      handlers: createHandlers(),
    },
  },
};

export const ErrorState: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/agl/v1/rollouts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }, { status: 200 }),
        ),
        http.get('*/agl/v1/rollouts/:rolloutId/attempts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }, { status: 200 }),
        ),
        http.get('*/agl/v1/spans', () =>
          HttpResponse.json({ detail: 'server error' }, { status: 500 }),
        ),
      ],
    },
  },
};
