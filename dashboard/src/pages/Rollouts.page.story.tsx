// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { waitFor, within } from '@testing-library/dom';
import userEvent from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';
import { Provider } from 'react-redux';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { AppAlertBanner } from '@/components/AppAlertBanner';
import { AppDrawerContainer } from '@/components/AppDrawer.component';
import { AppLayout } from '@/layouts/AppLayout';
import { createMockHandlers } from '@/utils/mock';
import { STORY_BASE_URL, STORY_DATE_NOW_SECONDS } from '../../.storybook/constants';
import { allModes } from '../../.storybook/modes';
import { initialConfigState } from '../features/config/slice';
import { initialResourcesUiState } from '../features/resources/slice';
import type { Attempt, Rollout, Span } from '../features/rollouts';
import { initialRolloutsUiState, type RolloutsUiState } from '../features/rollouts/slice';
import { createAppStore } from '../store';
import { RolloutsPage } from './Rollouts.page';

const meta: Meta<typeof RolloutsPage> = {
  title: 'Pages/RolloutsPage',
  component: RolloutsPage,
  parameters: {
    layout: 'fullscreen',
    chromatic: {
      modes: allModes,
    },
  },
};

export default meta;

type Story = StoryObj<typeof RolloutsPage>;

const now = STORY_DATE_NOW_SECONDS;

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

const sampleSpansByAttempt: Record<string, Span[]> = {
  'ro-7fa3b6e2:at-9001': [
    {
      rolloutId: 'ro-7fa3b6e2',
      attemptId: 'at-9001',
      sequenceId: 1,
      traceId: 'tr-7fa3b6e2-1',
      spanId: 'sp-7fa3b6e2-setup',
      parentId: null,
      name: 'Initialize rollout',
      status: { status_code: 'OK', description: null },
      attributes: { step: 'init', duration_ms: 120 },
      startTime: now - 1100,
      endTime: now - 1000,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
    {
      rolloutId: 'ro-7fa3b6e2',
      attemptId: 'at-9001',
      sequenceId: 2,
      traceId: 'tr-7fa3b6e2-1',
      spanId: 'sp-7fa3b6e2-run',
      parentId: 'sp-7fa3b6e2-setup',
      name: 'Execute task',
      status: { status_code: 'OK', description: null },
      attributes: { step: 'run', duration_ms: 450 },
      startTime: now - 950,
      endTime: now - 500,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ],
  'ro-116eab45:at-9002': [
    {
      rolloutId: 'ro-116eab45',
      attemptId: 'at-9002',
      sequenceId: 1,
      traceId: 'tr-116eab45-1',
      spanId: 'sp-116eab45-validate',
      parentId: null,
      name: 'Validate input',
      status: { status_code: 'OK', description: null },
      attributes: { step: 'validate', duration_ms: 80 },
      startTime: now - 3800,
      endTime: now - 3720,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
    {
      rolloutId: 'ro-116eab45',
      attemptId: 'at-9002',
      sequenceId: 2,
      traceId: 'tr-116eab45-1',
      spanId: 'sp-116eab45-execute',
      parentId: 'sp-116eab45-validate',
      name: 'Execute workflow',
      status: { status_code: 'OK', description: null },
      attributes: { step: 'execute', duration_ms: 260 },
      startTime: now - 3700,
      endTime: now - 3440,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ],
  'ro-9ae77c11:at-9005': [
    {
      rolloutId: 'ro-9ae77c11',
      attemptId: 'at-9005',
      sequenceId: 1,
      traceId: 'tr-9ae77c11-1',
      spanId: 'sp-9ae77c11-fetch',
      parentId: null,
      name: 'Fetch resources',
      status: { status_code: 'OK', description: null },
      attributes: { step: 'fetch', duration_ms: 200 },
      startTime: now - 8700,
      endTime: now - 8500,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
    {
      rolloutId: 'ro-9ae77c11',
      attemptId: 'at-9005',
      sequenceId: 2,
      traceId: 'tr-9ae77c11-1',
      spanId: 'sp-9ae77c11-run',
      parentId: 'sp-9ae77c11-fetch',
      name: 'Run evaluation',
      status: { status_code: 'ERROR', description: 'Worker timeout' },
      attributes: { step: 'evaluate', duration_ms: 600 },
      startTime: now - 8450,
      endTime: now - 7850,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ],
};

const overflowDrawerSpans: Span[] = Array.from({ length: 160 }, (_, index) => ({
  rolloutId: 'ro-7fa3b6e2',
  attemptId: 'at-9001',
  sequenceId: index + 1,
  traceId: `tr-overflow-${Math.floor(index / 5)}`,
  spanId: `sp-overflow-${index + 1}`,
  parentId: index === 0 ? null : `sp-overflow-${index}`,
  name: `Overflow span ${index + 1}`,
  status: { status_code: 'OK', description: null },
  attributes: { step: `overflow-${index + 1}`, duration_ms: 20 + (index % 5) },
  startTime: now - 1_200 - index * 20,
  endTime: now - 1_180 - index * 20,
  events: [],
  links: [],
  context: {},
  parent: null,
  resource: {},
}));

const overflowSpansByAttempt: Record<string, Span[]> = {
  ...sampleSpansByAttempt,
  'ro-7fa3b6e2:at-9001': overflowDrawerSpans,
};

const longJsonLogs = Array.from({ length: 200 }, (_, index) => ({
  id: index + 1,
  detail: `Log entry ${index + 1} ${'x'.repeat(32)}`,
  timestamp: now - index * 2,
}));

const jsonOverflowAttempt: Attempt = {
  rolloutId: 'ro-json-overflow',
  attemptId: 'at-json-overflow',
  sequenceId: 1,
  status: 'succeeded',
  startTime: now - 600,
  endTime: now - 300,
  workerId: 'worker-scroll',
  lastHeartbeatTime: now - 300,
  metadata: { notes: 'Completed with a very large JSON payload' },
};

const jsonOverflowAttemptEndTime = jsonOverflowAttempt.endTime ?? jsonOverflowAttempt.startTime + 1;

const jsonOverflowRollout: Rollout = {
  rolloutId: 'ro-json-overflow',
  input: {
    task: 'Render large JSON',
    payload: longJsonLogs,
    summary: 'This rollout includes many log lines to test scroll behavior.',
  },
  status: 'succeeded',
  mode: 'train',
  resourcesId: 'rs-json-overflow',
  startTime: jsonOverflowAttempt.startTime,
  endTime: jsonOverflowAttemptEndTime,
  attempt: jsonOverflowAttempt,
  config: {
    retries: 0,
    parameters: { max_steps: 200, batch: 5 },
  },
  metadata: {
    owner: 'scroll-tester',
    description: 'Synthetic rollout with oversized JSON payload for storybook validation.',
    tags: Array.from({ length: 40 }, (_, index) => `tag-${index + 1}`),
  },
};

const jsonOverflowSpans: Span[] = [
  {
    rolloutId: jsonOverflowRollout.rolloutId,
    attemptId: jsonOverflowAttempt.attemptId,
    sequenceId: 1,
    traceId: 'tr-json-overflow',
    spanId: 'sp-json-root',
    parentId: null,
    name: 'json-overflow-root',
    status: { status_code: 'OK', description: null },
    attributes: { detail: 'root span' },
    startTime: jsonOverflowAttempt.startTime,
    endTime: jsonOverflowAttemptEndTime,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
];

const jsonOverflowRollouts = [jsonOverflowRollout, ...sampleRollouts];
const jsonOverflowAttemptsByRollout: Record<string, Attempt[]> = {
  ...attemptsByRollout,
  [jsonOverflowRollout.rolloutId]: [jsonOverflowAttempt],
};
const jsonOverflowSpansByAttempt: Record<string, Span[]> = {
  ...sampleSpansByAttempt,
  [`${jsonOverflowRollout.rolloutId}:${jsonOverflowAttempt.attemptId}`]: jsonOverflowSpans,
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
  paginationRollouts.map((rollout) => [rollout.rolloutId, rollout.attempt ? [rollout.attempt] : []]),
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

function createStoryStore(
  uiOverrides?: Partial<RolloutsUiState>,
  configOverrides?: Partial<typeof initialConfigState>,
) {
  return createAppStore({
    config: {
      ...initialConfigState,
      baseUrl: STORY_BASE_URL,
      autoRefreshMs: 0,
      ...configOverrides,
    },
    rollouts: {
      ...initialRolloutsUiState,
      ...uiOverrides,
    },
    resources: initialResourcesUiState,
  });
}

function renderWithStore(uiOverrides?: Partial<RolloutsUiState>, configOverrides?: Partial<typeof initialConfigState>) {
  const store = createStoryStore(uiOverrides, configOverrides);

  return (
    <Provider store={store}>
      <>
        <RolloutsPage />
        <AppAlertBanner />
        <AppDrawerContainer />
      </>
    </Provider>
  );
}

function renderWithAppLayout(
  uiOverrides?: Partial<RolloutsUiState>,
  configOverrides?: Partial<typeof initialConfigState>,
) {
  const store = createStoryStore(uiOverrides, configOverrides);
  const router = createMemoryRouter(
    [
      {
        path: '/',
        element: (
          <AppLayout
            config={{
              baseUrl: store.getState().config.baseUrl,
              autoRefreshMs: store.getState().config.autoRefreshMs,
            }}
          />
        ),
        children: [
          {
            path: '/rollouts',
            element: <RolloutsPage />,
          },
        ],
      },
    ],
    { initialEntries: ['/rollouts'] },
  );

  return (
    <Provider store={store}>
      <>
        <RouterProvider router={router} />
        <AppDrawerContainer />
      </>
    </Provider>
  );
}

const defaultHandlers = createMockHandlers(sampleRollouts, attemptsByRollout, sampleSpansByAttempt);
const overflowHandlers = createMockHandlers(sampleRollouts, attemptsByRollout, overflowSpansByAttempt);
const jsonOverflowHandlers = createMockHandlers(
  jsonOverflowRollouts,
  jsonOverflowAttemptsByRollout,
  jsonOverflowSpansByAttempt,
);

export const Default: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
};

export const WithSidebarLayout: Story = {
  name: 'Within AppLayout',
  render: () => renderWithAppLayout(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
};

export const WithSidebarStatusFilter: Story = {
  name: 'Within AppLayout (Status Filter)',
  render: () => renderWithAppLayout({ statusFilters: ['running'] }),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async () => {
    await waitFor(() => {
      const container = document.querySelector<HTMLElement>('[data-testid="rollouts-table-container"]');
      const main = document.querySelector<HTMLElement>('.mantine-AppShell-main');
      if (!container || !main) {
        throw new Error('Unable to locate rollout table container or AppShell main region');
      }
      const containerRect = container.getBoundingClientRect();
      const mainRect = main.getBoundingClientRect();
      if (containerRect.right > mainRect.right + 1) {
        throw new Error('Rollouts table extends beyond the AppShell content area');
      }
    });
  },
};

export const DarkTheme: Story = {
  render: () => renderWithStore(undefined, { theme: 'dark' }),
  parameters: {
    theme: 'dark',
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
        http.get('*/v1/agl/rollouts', () => HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 })),
        http.get('*/v1/agl/rollouts/:rolloutId/attempts', () =>
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
        http.get('*/v1/agl/rollouts', () => HttpResponse.json({ detail: 'Internal error' }, { status: 500 })),
        http.get('*/v1/agl/rollouts/:rolloutId/attempts', () =>
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
        http.get('*/v1/agl/rollouts', async () => {
          await delay('infinite');
          return HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 });
        }),
        http.get('*/v1/agl/rollouts/:rolloutId/attempts', async () => {
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
      handlers: createMockHandlers(longDurationRollouts, longDurationAttempts, {}),
    },
  },
};

export const StaleHeartbeat: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(staleHeartbeatRollouts, staleHeartbeatAttempts, {}),
    },
  },
};

export const StatusMismatch: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(statusMismatchRollouts, statusMismatchAttempts, {}),
    },
  },
};

export const LongInput: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(longInputRollouts, longInputAttempts, {}),
    },
  },
};

export const Pagination: Story = {
  render: () => renderWithStore({ recordsPerPage: 20 }),
  parameters: {
    msw: {
      handlers: createMockHandlers(paginationRollouts, paginationAttempts, {}),
    },
  },
};

export const AutoExpandedAttempt: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createMockHandlers(autoExpandRollouts, autoExpandAttempts, {}),
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
    await waitFor(
      async () => {
        await canvas.findByText('at-expand-001');
      },
      { timeout: 3_000 },
    );
  },
};

export const Search: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('ro-7fa3b6e2');

    const searchInput = canvas.getByPlaceholderText('Search by Rollout ID');
    await userEvent.type(searchInput, 'ro-116eab45');

    await waitFor(() => {
      if (canvas.queryByText('ro-7fa3b6e2')) {
        throw new Error('Expected search to filter out non-matching rollouts');
      }
      if (!canvas.queryByText('ro-116eab45')) {
        throw new Error('Expected search to keep the matching rollout visible');
      }
    });
  },
};

async function openSampleTracesDrawer(canvasElement: HTMLElement) {
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

  return within(document.body).findByRole('dialog');
}

async function openRawJsonDrawer(canvasElement: HTMLElement, rolloutId = 'ro-7fa3b6e2') {
  const canvas = within(canvasElement);
  await canvas.findByText(rolloutId);
  const rolloutCell = canvas.getByText(rolloutId);
  const rolloutRow = rolloutCell.closest('tr');

  if (!rolloutRow) {
    throw new Error(`Unable to locate rollout row for ${rolloutId}`);
  }

  const rowScope = within(rolloutRow);
  const rawButtons = rowScope.getAllByRole('button', { name: 'View raw JSON' });
  const rawButton = rawButtons[0];
  await userEvent.click(rawButton);

  return within(document.body).findByRole('dialog');
}

export const RawJsonDrawer: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const drawer = await openRawJsonDrawer(canvasElement);
    await waitFor(
      async () => {
        await within(drawer).findByText('Attempt');
        await within(drawer).findByText(/worker-alpha/);
      },
      { timeout: 3_000 },
    );
  },
};

export const RawJsonDrawerScrollable: Story = {
  name: 'Raw JSON Drawer Scrollable',
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: jsonOverflowHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const drawer = await openRawJsonDrawer(canvasElement, 'ro-json-overflow');
    const editorContainer = drawer.querySelector('[data-testid="json-editor-container"]') as HTMLElement | null;
    if (!editorContainer) {
      throw new Error('Unable to locate JSON editor container');
    }
    await waitFor(() => {
      const scrollable = editorContainer.querySelector('.monaco-scrollable-element') as HTMLElement | null;
      if (!scrollable) {
        throw new Error('Monaco editor not ready yet');
      }
      if (scrollable.scrollHeight <= scrollable.clientHeight) {
        throw new Error('Expected JSON content to overflow and allow scrolling');
      }
    });
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
    await openSampleTracesDrawer(canvasElement);
  },
};

export const TracesDrawerLink: Story = {
  name: 'Traces Drawer Link',
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const drawer = await openSampleTracesDrawer(canvasElement);
    const link = await within(drawer).findByText('View full traces');
    const href = link.getAttribute('href');
    if (!href) {
      throw new Error('Expected traces drawer to render a link to the traces page');
    }
    if (!href.includes('rolloutId=ro-7fa3b6e2')) {
      throw new Error(`Link href ${href} is missing rolloutId query parameter`);
    }
    if (!href.includes('attemptId=at-9001')) {
      throw new Error(`Link href ${href} is missing attemptId query parameter`);
    }
  },
};

export const TracesDrawerScrollableTable: Story = {
  name: 'Traces Drawer Scrollable Table',
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: overflowHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const drawer = await openSampleTracesDrawer(canvasElement);
    const container = drawer.querySelector('[data-testid="traces-drawer-table-container"]') as HTMLElement | null;
    if (!container) {
      throw new Error('Unable to locate traces table container inside drawer');
    }
    const overflowStyle = window.getComputedStyle(container).overflowY;
    if (overflowStyle !== 'auto' && overflowStyle !== 'scroll') {
      throw new Error('Expected traces table container to allow vertical scrolling');
    }
    await waitFor(() => {
      if (container.scrollHeight <= container.clientHeight) {
        throw new Error('Expected traces table content to overflow and enable scrolling');
      }
    });
  },
};
