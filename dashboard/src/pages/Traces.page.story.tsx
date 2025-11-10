// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { delay, http, HttpResponse } from 'msw';
import { Provider } from 'react-redux';
import { AppAlertBanner } from '@/components/AppAlertBanner';
import { AppDrawerContainer } from '@/components/AppDrawer.component';
import { createMockHandlers } from '@/utils/mock';
import { allModes } from '../../.storybook/modes';
import { initialConfigState } from '../features/config/slice';
import { initialResourcesUiState } from '../features/resources/slice';
import { initialRolloutsUiState } from '../features/rollouts/slice';
import { initialTracesUiState, type TracesUiState } from '../features/traces/slice';
import { createAppStore } from '../store';
import type { Attempt, Rollout, Span } from '../types';
import { TracesPage } from './Traces.page';

const meta: Meta<typeof TracesPage> = {
  title: 'Pages/TracesPage',
  component: TracesPage,
  parameters: {
    layout: 'fullscreen',
    chromatic: {
      modes: allModes,
    },
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

const emptyAttemptsByRollout: Record<string, Attempt[]> = {};
const emptySpansByAttempt: Record<string, Span[]> = {};

const singleRollout = sampleRollouts[1];
const singleRollouts: Rollout[] = [singleRollout];
const singleAttemptsByRollout: Record<string, Attempt[]> = {
  [singleRollout.rolloutId]: attemptsByRollout[singleRollout.rolloutId] ?? [],
};
const singleSpansByAttempt = Object.fromEntries(
  Object.entries(spansByAttempt).filter(([key]) => key.startsWith(`${singleRollout.rolloutId}:`)),
) as Record<string, Span[]>;

const owners = ['ava', 'ben', 'carla', 'diego'] as const;

const manyAttemptsByRollout: Record<string, Attempt[]> = {};
const manySpansByAttempt: Record<string, Span[]> = {};

const manyRollouts: Rollout[] = Array.from({ length: 24 }, (_, index) => {
  const rolloutId = `ro-many-${String(index + 1).padStart(3, '0')}`;
  const statusOptions = ['running', 'succeeded', 'failed'] as const;
  const modeOptions = ['train', 'val', 'test'] as const;
  const status = statusOptions[index % statusOptions.length];
  const mode = modeOptions[index % modeOptions.length];
  const startTime = now - (index + 1) * 420;
  const endTime = status === 'running' ? null : startTime + 240;
  const attemptId = `${rolloutId}-attempt`;
  const attemptStatus: Attempt['status'] =
    status === 'failed' ? 'failed' : status === 'succeeded' ? 'succeeded' : 'running';
  const attempt: Attempt = {
    rolloutId,
    attemptId,
    sequenceId: 1,
    status: attemptStatus,
    startTime,
    endTime,
    workerId: `worker-${String.fromCharCode(97 + (index % 26))}`,
    lastHeartbeatTime: endTime ?? startTime + 180,
    metadata: { region: index % 2 === 0 ? 'us-east-1' : 'eu-west-1' },
  };
  manyAttemptsByRollout[rolloutId] = [attempt];
  manySpansByAttempt[`${rolloutId}:${attemptId}`] = [
    {
      rolloutId,
      attemptId,
      sequenceId: 1,
      traceId: `tr-many-${index + 1}`,
      spanId: `sp-many-${index + 1}-root`,
      parentId: null,
      name: 'Synthetic root span',
      status: {
        status_code: status === 'failed' ? 'ERROR' : 'OK',
        description: status === 'failed' ? 'Synthetic failure' : null,
      },
      attributes: {
        'trace.sample': index + 1,
        'duration_ms': 240,
      },
      startTime,
      endTime: endTime ?? startTime + 240,
      events: [],
      links: [],
      context: {},
      parent: null,
      resource: {},
    },
  ];
  return {
    rolloutId,
    input: { task: `Synthetic trace ${index + 1}` },
    status,
    mode,
    resourcesId: `rs-many-${(index % 7) + 1}`,
    startTime,
    endTime,
    attempt,
    config: { retries: index % 3 },
    metadata: { owner: owners[index % owners.length] },
  };
});

function createHandlers(delayMs?: number) {
  return createMockHandlers(sampleRollouts, attemptsByRollout, spansByAttempt, delayMs);
}

function createRequestTimeoutHandlers() {
  return [
    http.get('*/v1/agl/rollouts', async () => {
      await delay(1200);
      return HttpResponse.json({ detail: 'Request timed out' }, { status: 504, statusText: 'Timeout' });
    }),
    http.get('*/v1/agl/rollouts/:rolloutId/attempts', async ({ params }) => {
      await delay(1200);
      return HttpResponse.json(
        { detail: 'Request timed out', rolloutId: params.rolloutId },
        { status: 504, statusText: 'Timeout' },
      );
    }),
    http.get('*/v1/agl/spans', async () => {
      await delay(1200);
      return HttpResponse.json({ detail: 'Request timed out' }, { status: 504, statusText: 'Timeout' });
    }),
  ];
}

const rolloutsAndAttemptsHandlers = createMockHandlers(sampleRollouts, attemptsByRollout);

function renderTracesPage(
  preloadedTracesState?: Partial<TracesUiState>,
  configOverrides?: Partial<typeof initialConfigState>,
) {
  const store = createAppStore({
    config: {
      ...initialConfigState,
      ...configOverrides,
    },
    rollouts: initialRolloutsUiState,
    resources: initialResourcesUiState,
    traces: { ...initialTracesUiState, ...preloadedTracesState },
  });

  return (
    <Provider store={store}>
      <TracesPage />
      <AppAlertBanner />
      <AppDrawerContainer />
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

export const DarkTheme: Story = {
  render: () => renderTracesPage(undefined, { theme: 'dark' }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: createHandlers(),
    },
  },
};

export const EmptyState: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers([], emptyAttemptsByRollout, emptySpansByAttempt),
    },
  },
};

export const SingleResult: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers(singleRollouts, singleAttemptsByRollout, singleSpansByAttempt),
    },
  },
};

export const ManyResults: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createMockHandlers(manyRollouts, manyAttemptsByRollout, manySpansByAttempt),
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

export const RequestTimeout: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: createRequestTimeoutHandlers(),
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

export const ServerError: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/v1/agl/rollouts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }, { status: 200 }),
        ),
        http.get('*/v1/agl/rollouts/:rolloutId/attempts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }, { status: 200 }),
        ),
        http.get('*/v1/agl/spans', () => HttpResponse.json({ detail: 'server error' }, { status: 500 })),
      ],
    },
  },
};

export const RolloutParseFailure: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/v1/agl/rollouts', () =>
          HttpResponse.text('not valid json', {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
            },
          }),
        ),
        http.get('*/v1/agl/rollouts/:rolloutId/attempts', () =>
          HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }),
        ),
        http.get('*/v1/agl/spans', () => HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 })),
      ],
    },
  },
};

export const ParseFailure: Story = {
  render: () => renderTracesPage(),
  parameters: {
    msw: {
      handlers: [
        ...rolloutsAndAttemptsHandlers,
        http.get('*/v1/agl/spans', () =>
          HttpResponse.text('not valid json', {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
            },
          }),
        ),
      ],
    },
  },
};
