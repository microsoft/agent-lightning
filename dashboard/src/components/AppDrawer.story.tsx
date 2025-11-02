import type { Meta, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { AppDrawer } from './AppDrawer.component';
import { createAppStore } from '@/store';
import { initialConfigState } from '@/features/config/slice';
import { initialRolloutsUiState } from '@/features/rollouts/slice';
import { initialTracesUiState } from '@/features/traces/slice';
import { initialResourcesUiState } from '@/features/resources/slice';
import type { Attempt, Rollout, Span } from '@/types';
import type { DrawerContent } from '@/features/ui/drawer';

const meta = {
  title: 'Components/AppDrawer',
  component: AppDrawer,
  parameters: {
    layout: 'fullscreen',
  },
} satisfies Meta<typeof AppDrawer>;

export default meta;

type Story = StoryObj<typeof AppDrawer>;

const now = Math.floor(Date.now() / 1000);

const baseAttempt: Attempt = {
  rolloutId: 'ro-story-001',
  attemptId: 'at-story-001',
  sequenceId: 1,
  startTime: now - 3600,
  endTime: null,
  status: 'running',
  workerId: 'worker-story',
  lastHeartbeatTime: now - 42,
  metadata: { info: 'Sample metadata', runId: 'run-123' },
};

const baseRollout: Rollout = {
  rolloutId: 'ro-story-001',
  input: {
    task: 'Generate daily summary',
    payload: { account: 'enterprise', date: '2024-02-19' },
  },
  startTime: now - 4000,
  endTime: null,
  mode: 'train',
  resourcesId: 'rs-story-001',
  status: 'running',
  config: { retries: 1, priority: 'high' },
  metadata: { owner: 'storybook' },
  attempt: baseAttempt,
};

const noAttemptRollout: Rollout = {
  ...baseRollout,
  status: 'queuing',
  attempt: null,
};

const mismatchRollout: Rollout = {
  ...baseRollout,
  status: 'running',
  attempt: {
    ...baseAttempt,
    status: 'failed',
    endTime: now - 1200,
    metadata: { info: 'Latest attempt failed', reason: 'Timeout' },
  },
};

const sampleSpan: Span = {
  rolloutId: 'ro-story-001',
  attemptId: 'at-story-001',
  sequenceId: 2,
  traceId: 'tr-story-001',
  spanId: 'sp-story-001',
  parentId: null,
  name: 'Fetch Resources',
  status: { status_code: 'OK', description: 'Completed successfully' },
  attributes: {
    'http.method': 'GET',
    'http.url': 'https://api.example.com/resources',
    'duration_ms': 120,
  },
  startTime: now - 240,
  endTime: now - 120,
  events: [],
  links: [],
  context: {},
  parent: null,
  resource: {},
};

function renderWithDrawer(content: DrawerContent) {
  const store = createAppStore({
    config: initialConfigState,
    rollouts: initialRolloutsUiState,
    resources: initialResourcesUiState,
    traces: initialTracesUiState,
    drawer: {
      isOpen: true,
      content,
    },
  });

  return (
    <Provider store={store}>
      <AppDrawer />
    </Provider>
  );
}

export const RolloutJson: Story = {
  render: () =>
    renderWithDrawer({
      type: 'rollout-json',
      rollout: baseRollout,
      attempt: baseRollout.attempt,
      isNested: false,
    }),
};

export const NestedAttemptJson: Story = {
  render: () =>
    renderWithDrawer({
      type: 'rollout-json',
      rollout: baseRollout,
      attempt: {
        ...baseAttempt,
        attemptId: 'at-story-002',
        sequenceId: 2,
        status: 'failed',
        endTime: now - 1200,
        metadata: { info: 'Secondary attempt', reason: 'Timeout' },
      },
      isNested: true,
    }),
};

export const TracesPlaceholder: Story = {
  render: () =>
    renderWithDrawer({
      type: 'rollout-traces',
      rollout: baseRollout,
      attempt: baseRollout.attempt,
      isNested: false,
    }),
};

export const NoAttempt: Story = {
  render: () =>
    renderWithDrawer({
      type: 'rollout-json',
      rollout: noAttemptRollout,
      attempt: null,
      isNested: false,
    }),
};

export const StatusMismatch: Story = {
  render: () =>
    renderWithDrawer({
      type: 'rollout-json',
      rollout: mismatchRollout,
      attempt: mismatchRollout.attempt,
      isNested: false,
    }),
};

export const SpanDetail: Story = {
  render: () =>
    renderWithDrawer({
      type: 'trace-detail',
      span: sampleSpan,
      rollout: mismatchRollout,
      attempt: mismatchRollout.attempt,
    }),
};

export const LightTheme: Story = {
  render: () =>
    renderWithDrawer({
      type: 'rollout-json',
      rollout: baseRollout,
      attempt: baseRollout.attempt,
      isNested: false,
    }),
  parameters: {
    theme: 'light',
  },
};

export const DarkTheme: Story = {
  render: () =>
    renderWithDrawer({
      type: 'trace-detail',
      span: {
        ...sampleSpan,
        spanId: 'sp-story-002',
        name: 'Process Response',
        status: { status_code: 'ERROR', description: 'Unexpected response code' },
      },
      rollout: mismatchRollout,
      attempt: mismatchRollout.attempt,
    }),
  parameters: {
    theme: 'dark',
  },
};
