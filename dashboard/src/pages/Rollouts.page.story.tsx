import type { Meta, StoryObj } from '@storybook/react';
import userEvent from '@testing-library/user-event';
import { within } from '@testing-library/dom';
import { Provider } from 'react-redux';
import { http, HttpResponse, delay } from 'msw';
import { RolloutsPage } from './Rollouts.page';
import { AppDrawer } from '@/components/AppDrawer.component';
import { createAppStore } from '../store';
import { initialConfigState } from '../features/config/slice';
import { initialRolloutsUiState, type RolloutsUiState } from '../features/rollouts/slice';
import { initialResourcesUiState } from '../features/resources/slice';
import type { Attempt, Rollout } from '../features/rollouts';
import { snakeCaseKeys } from '@/utils/format';

const meta: Meta<typeof RolloutsPage> = {
  title: 'Pages/RolloutsPage',
  component: RolloutsPage,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof RolloutsPage>;

const now = Math.floor(Date.now() / 1000);

const sampleRollouts: Rollout[] = [
  {
    rolloutId: 'ro-7fa3b6e2',
    input: { task: 'Summarize report' },
    status: 'running',
    mode: 'train',
    resourcesId: 'rs-100',
    startTime: now - 1200,
    endTime: null,
    attempt: {
      rolloutId: 'ro-7fa3b6e2',
      attemptId: 'at-9001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 1200,
      endTime: null,
      workerId: 'worker-alpha',
      lastHeartbeatTime: now - 30,
      metadata: { lastHeartbeatAt: now - 30 },
    },
    config: { retries: 0 },
    metadata: { owner: 'alice' },
  },
  {
    rolloutId: 'ro-116eab45',
    input: { task: 'Classify dataset' },
    status: 'succeeded',
    mode: 'val',
    resourcesId: 'rs-101',
    startTime: now - 5400,
    endTime: now - 3600,
    attempt: {
      rolloutId: 'ro-116eab45',
      attemptId: 'at-9002',
      sequenceId: 2,
      status: 'succeeded',
      startTime: now - 4000,
      endTime: now - 3600,
      workerId: 'worker-beta',
      lastHeartbeatTime: now - 3600,
      metadata: { lastHeartbeatAt: now - 3600 },
    },
    config: { retries: 1 },
    metadata: { owner: 'bob' },
  },
  {
    rolloutId: 'ro-9ae77c11',
    input: { task: 'Evaluate prompt variations' },
    status: 'failed',
    mode: 'test',
    resourcesId: 'rs-102',
    startTime: now - 9600,
    endTime: now - 8400,
    attempt: {
      rolloutId: 'ro-9ae77c11',
      attemptId: 'at-9005',
      sequenceId: 3,
      status: 'failed',
      startTime: now - 8800,
      endTime: now - 8400,
      workerId: 'worker-gamma',
      lastHeartbeatTime: now - 8400,
      metadata: { lastHeartbeatAt: now - 8400 },
    },
    config: { retries: 2 },
    metadata: { owner: 'carol' },
  },
];

const attemptsByRollout: Record<string, Attempt[]> = {
  'ro-7fa3b6e2': [
    {
      rolloutId: 'ro-7fa3b6e2',
      attemptId: 'at-9001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 1200,
      endTime: null,
      workerId: 'worker-alpha',
      lastHeartbeatTime: now - 30,
      metadata: { lastHeartbeatAt: now - 30 },
    },
  ],
  'ro-116eab45': [
    {
      rolloutId: 'ro-116eab45',
      attemptId: 'at-9000',
      sequenceId: 1,
      status: 'failed',
      startTime: now - 5400,
      endTime: now - 5000,
      workerId: 'worker-beta',
      lastHeartbeatTime: now - 5000,
      metadata: { lastHeartbeatAt: now - 5000 },
    },
    {
      rolloutId: 'ro-116eab45',
      attemptId: 'at-9002',
      sequenceId: 2,
      status: 'succeeded',
      startTime: now - 4000,
      endTime: now - 3600,
      workerId: 'worker-beta',
      lastHeartbeatTime: now - 3600,
      metadata: { lastHeartbeatAt: now - 3600 },
    },
  ],
  'ro-9ae77c11': [
    {
      rolloutId: 'ro-9ae77c11',
      attemptId: 'at-9003',
      sequenceId: 1,
      status: 'preparing',
      startTime: now - 9600,
      endTime: now - 9300,
      workerId: 'worker-gamma',
      lastHeartbeatTime: now - 9300,
      metadata: { lastHeartbeatAt: now - 9300 },
    },
    {
      rolloutId: 'ro-9ae77c11',
      attemptId: 'at-9004',
      sequenceId: 2,
      status: 'running',
      startTime: now - 9200,
      endTime: now - 8800,
      workerId: 'worker-delta',
      lastHeartbeatTime: now - 8800,
      metadata: { lastHeartbeatAt: now - 8800 },
    },
    {
      rolloutId: 'ro-9ae77c11',
      attemptId: 'at-9005',
      sequenceId: 3,
      status: 'failed',
      startTime: now - 8800,
      endTime: now - 8400,
      workerId: 'worker-gamma',
      lastHeartbeatTime: now - 8400,
      metadata: { lastHeartbeatAt: now - 8400 },
    },
  ],
};

const longDurationRollouts: Rollout[] = [
  {
    rolloutId: 'ro-long-duration',
    input: { task: 'Long running training' },
    status: 'running',
    mode: 'train',
    resourcesId: 'rs-200',
    startTime: now - 7 * 24 * 3600,
    endTime: null,
    attempt: {
      rolloutId: 'ro-long-duration',
      attemptId: 'at-long-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 7 * 24 * 3600,
      endTime: null,
      workerId: 'worker-long',
      lastHeartbeatTime: now - 45,
      metadata: null,
    },
    config: { retries: 0 },
    metadata: { owner: 'delta' },
  },
];

const longDurationAttempts: Record<string, Attempt[]> = {
  'ro-long-duration': [
    {
      rolloutId: 'ro-long-duration',
      attemptId: 'at-long-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 7 * 24 * 3600,
      endTime: null,
      workerId: 'worker-long',
      lastHeartbeatTime: now - 45,
      metadata: null,
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

const buildRolloutsResponse = (rollouts: Rollout[], request: Request) => {
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
  attemptList: Attempt[],
  sortBy: string | null,
  sortOrder: 'asc' | 'desc',
): Attempt[] => {
  const sorted = [...attemptList];
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

const buildAttemptsResponse = (attemptList: Attempt[], request: Request) => {
  const url = new URL(request.url);
  const params = url.searchParams;
  const sortBy = params.get('sort_by');
  const sortOrder = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorted = sortAttemptsForParams(attemptList, sortBy, sortOrder);
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
    total: attemptList.length,
  });
};

const staleHeartbeatRollouts: Rollout[] = [
  {
    rolloutId: 'ro-stale-heartbeat',
    input: { task: 'Investigate stale worker' },
    status: 'running',
    mode: 'test',
    resourcesId: 'rs-201',
    startTime: now - 6 * 3600,
    endTime: null,
    attempt: {
      rolloutId: 'ro-stale-heartbeat',
      attemptId: 'at-stale-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 6 * 3600,
      endTime: null,
      workerId: 'worker-stale',
      lastHeartbeatTime: now - 3 * 24 * 3600,
      metadata: null,
    },
    config: { retries: 0 },
    metadata: { owner: 'echo' },
  },
];

const staleHeartbeatAttempts: Record<string, Attempt[]> = {
  'ro-stale-heartbeat': [
    {
      rolloutId: 'ro-stale-heartbeat',
      attemptId: 'at-stale-001',
      sequenceId: 1,
      status: 'running',
      startTime: now - 6 * 3600,
      endTime: null,
      workerId: 'worker-stale',
      lastHeartbeatTime: now - 3 * 24 * 3600,
      metadata: null,
    },
  ],
};

const statusMismatchRollouts: Rollout[] = [
  {
    rolloutId: 'ro-status-mismatch',
    input: { task: 'Edge case validation' },
    status: 'running',
    mode: 'val',
    resourcesId: null,
    startTime: now - 3600,
    endTime: now - 1800,
    attempt: {
      rolloutId: 'ro-status-mismatch',
      attemptId: 'at-mismatch-003',
      sequenceId: 3,
      status: 'failed',
      startTime: now - 4200,
      endTime: now - 1800,
      workerId: 'worker-mismatch',
      lastHeartbeatTime: now - 1700,
      metadata: null,
    },
    config: { retries: 3 },
    metadata: { owner: 'foxtrot' },
  },
];

const statusMismatchAttempts: Record<string, Attempt[]> = {
  'ro-status-mismatch': [
    {
      rolloutId: 'ro-status-mismatch',
      attemptId: 'at-mismatch-001',
      sequenceId: 1,
      status: 'preparing',
      startTime: now - 5400,
      endTime: now - 5000,
      workerId: 'worker-mismatch',
      lastHeartbeatTime: now - 5000,
      metadata: null,
    },
    {
      rolloutId: 'ro-status-mismatch',
      attemptId: 'at-mismatch-002',
      sequenceId: 2,
      status: 'running',
      startTime: now - 5000,
      endTime: now - 4200,
      workerId: 'worker-mismatch',
      lastHeartbeatTime: now - 4000,
      metadata: null,
    },
    {
      rolloutId: 'ro-status-mismatch',
      attemptId: 'at-mismatch-003',
      sequenceId: 3,
      status: 'failed',
      startTime: now - 4200,
      endTime: now - 1800,
      workerId: 'worker-mismatch',
      lastHeartbeatTime: now - 1700,
      metadata: null,
    },
  ],
};

const veryLongInput = `{"prompt":"${'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(12)}"}`;

const longInputRollouts: Rollout[] = [
  {
    rolloutId: 'ro-long-input',
    input: veryLongInput,
    status: 'queuing',
    mode: 'test',
    resourcesId: 'rs-300',
    startTime: now - 120,
    endTime: null,
    attempt: null,
    config: { retries: 0 },
    metadata: { owner: 'golf' },
  },
];

const longInputAttempts: Record<string, Attempt[]> = {
  'ro-long-input': [],
};

const paginationRollouts: Rollout[] = Array.from({ length: 120 }, (_item, index) => {
  const startOffset = index * 90;
  const rolloutId = `ro-page-${index + 1}`;

  return {
    rolloutId,
    input: { item: index + 1 },
    status: index % 3 === 0 ? 'running' : index % 3 === 1 ? 'failed' : 'succeeded',
    mode: index % 2 === 0 ? 'train' : 'test',
    resourcesId: index % 5 === 0 ? `rs-${100 + index}` : null,
    startTime: now - startOffset - 300,
    endTime: index % 3 === 0 ? null : now - startOffset,
    attempt: {
      rolloutId,
      attemptId: `at-page-${index + 1}`,
      sequenceId: 1,
      status: index % 3 === 0 ? 'running' : index % 3 === 1 ? 'failed' : 'succeeded',
      startTime: now - startOffset - 300,
      endTime: index % 3 === 0 ? null : now - startOffset,
      workerId: `worker-${(index % 7) + 1}`,
      lastHeartbeatTime: index % 3 === 0 ? now - startOffset - 60 : now - startOffset,
      metadata: null,
    },
    config: {},
    metadata: null,
  };
});

const paginationAttempts: Record<string, Attempt[]> = Object.fromEntries(
  paginationRollouts.map((rollout) => [
    rollout.rolloutId,
    rollout.attempt ? [rollout.attempt] : [],
  ]),
);

const autoExpandRollouts: Rollout[] = [
  {
    rolloutId: 'ro-auto-expand',
    input: { task: 'Auto expand test' },
    status: 'running',
    mode: 'train',
    resourcesId: 'rs-400',
    startTime: now - 3600,
    endTime: null,
    attempt: {
      rolloutId: 'ro-auto-expand',
      attemptId: 'at-expand-003',
      sequenceId: 3,
      status: 'running',
      startTime: now - 3600,
      endTime: null,
      workerId: 'worker-auto',
      lastHeartbeatTime: now - 30,
      metadata: null,
    },
    config: {},
    metadata: null,
  },
];

const autoExpandAttempts: Record<string, Attempt[]> = {
  'ro-auto-expand': [
    {
      rolloutId: 'ro-auto-expand',
      attemptId: 'at-expand-001',
      sequenceId: 1,
      status: 'preparing',
      startTime: now - 5400,
      endTime: now - 5000,
      workerId: 'worker-auto',
      lastHeartbeatTime: now - 5000,
      metadata: null,
    },
    {
      rolloutId: 'ro-auto-expand',
      attemptId: 'at-expand-002',
      sequenceId: 2,
      status: 'failed',
      startTime: now - 5000,
      endTime: now - 4200,
      workerId: 'worker-auto',
      lastHeartbeatTime: now - 4200,
      metadata: null,
    },
    {
      rolloutId: 'ro-auto-expand',
      attemptId: 'at-expand-003',
      sequenceId: 3,
      status: 'running',
      startTime: now - 3600,
      endTime: null,
      workerId: 'worker-auto',
      lastHeartbeatTime: now - 30,
      metadata: null,
    },
  ],
};
function renderWithStore(uiOverrides?: Partial<RolloutsUiState>) {
  const store = createAppStore({
    config: {
      ...initialConfigState,
      autoRefreshMs: 0,
    },
    rollouts: {
      ...initialRolloutsUiState,
      ...uiOverrides,
    },
    resources: initialResourcesUiState,
  });

  return (
    <Provider store={store}>
      <>
        <RolloutsPage />
        <AppDrawer />
      </>
    </Provider>
  );
}

const createHandlers = (rollouts: Rollout[], attempts: Record<string, Attempt[]>) => [
  http.get('*/agl/v1/rollouts', ({ request }) =>
    HttpResponse.json(buildRolloutsResponse(rollouts, request)),
  ),
  http.get('*/agl/v1/rollouts/:rolloutId/attempts', ({ params, request }) => {
    const rolloutId = params.rolloutId as string;
    const attemptList = attempts[rolloutId] ?? [];
    return HttpResponse.json(buildAttemptsResponse(attemptList, request));
  }),
];

const defaultHandlers = createHandlers(sampleRollouts, attemptsByRollout);

export const Default: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
};

export const EmptyState: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/agl/v1/rollouts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }),
        ),
        http.get('*/agl/v1/rollouts/:rolloutId/attempts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }),
        ),
      ],
    },
  },
};

export const ServerError: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/agl/v1/rollouts', () =>
          HttpResponse.json({ detail: 'Internal error' }, { status: 500 }),
        ),
        http.get('*/agl/v1/rollouts/:rolloutId/attempts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }, { status: 200 }),
        ),
      ],
    },
  },
};

export const Loading: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/agl/v1/rollouts', async () => {
          await delay('infinite');
          return HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 });
        }),
        http.get('*/agl/v1/rollouts/:rolloutId/attempts', async () => {
          await delay('infinite');
          return HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 });
        }),
      ],
    },
  },
};

export const LongDuration: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createHandlers(longDurationRollouts, longDurationAttempts),
    },
  },
};

export const StaleHeartbeat: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createHandlers(staleHeartbeatRollouts, staleHeartbeatAttempts),
    },
  },
};

export const StatusMismatch: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createHandlers(statusMismatchRollouts, statusMismatchAttempts),
    },
  },
};

export const LongInput: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createHandlers(longInputRollouts, longInputAttempts),
    },
  },
};

export const Pagination: Story = {
  render: () => renderWithStore({ recordsPerPage: 20 }),
  parameters: {
    msw: {
      handlers: createHandlers(paginationRollouts, paginationAttempts),
    },
  },
};

export const AutoExpandedAttempt: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createHandlers(autoExpandRollouts, autoExpandAttempts),
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('ro-auto-expand');

    const rolloutCell = canvas.getByText('ro-auto-expand');
    const rolloutRow = rolloutCell.closest('tr');

    if (!rolloutRow) {
      throw new Error('Unable to locate the rollout row for expansion');
    }

    await userEvent.click(rolloutRow);
    await canvas.findByText('at-expand-001');
  },
};

export const RawJsonDrawer: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('ro-7fa3b6e2');
    const rolloutCell = canvas.getByText('ro-7fa3b6e2');
    const rolloutRow = rolloutCell.closest('tr');

    if (!rolloutRow) {
      throw new Error('Unable to locate rollout row for raw JSON drawer');
    }

    const rowScope = within(rolloutRow);
    const rawButtons = rowScope.getAllByRole('button', { name: 'View raw JSON' });
    const rawButton = rawButtons[0];
    await userEvent.click(rawButton);

    const drawer = await within(document.body).findByRole('dialog', { name: 'ro-7fa3b6e2' });
    await within(drawer).findByText('Attempt');
    await within(drawer).findByText(/worker-alpha/);
  },
};

export const TracesDrawer: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('ro-7fa3b6e2');
    const rolloutCell = canvas.getByText('ro-7fa3b6e2');
    const rolloutRow = rolloutCell.closest('tr');

    if (!rolloutRow) {
      throw new Error('Unable to locate rollout row for traces drawer');
    }

    const rowScope = within(rolloutRow);
    const traceButtons = rowScope.getAllByRole('button', { name: 'View traces' });
    const tracesButton = traceButtons[0];
    await userEvent.click(tracesButton);

    await within(document.body).findByRole('dialog', { name: 'ro-7fa3b6e2' });
  },
};
