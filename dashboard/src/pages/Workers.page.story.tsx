// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { waitFor, within } from '@testing-library/dom';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { AppAlertBanner } from '@/components/AppAlertBanner';
import { AppDrawerContainer } from '@/components/AppDrawer.component';
import { initialConfigState } from '@/features/config/slice';
import { initialWorkersUiState } from '@/features/workers/slice';
import { AppLayout } from '@/layouts/AppLayout';
import { createAppStore } from '@/store';
import type { Worker } from '@/types';
import { createWorkersHandlers } from '@/utils/mock';
import { STORY_BASE_URL, STORY_DATE_NOW_SECONDS } from '../../.storybook/constants';
import { allModes } from '../../.storybook/modes';
import { WorkersPage } from './Workers.page';

const meta: Meta<typeof WorkersPage> = {
  title: 'Pages/WorkersPage',
  component: WorkersPage,
  parameters: {
    layout: 'fullscreen',
    chromatic: {
      modes: allModes,
    },
  },
};

export default meta;

type Story = StoryObj<typeof WorkersPage>;

const now = STORY_DATE_NOW_SECONDS;

const sampleWorkers: Worker[] = [
  {
    workerId: 'worker-east',
    status: 'busy',
    heartbeatStats: { queueDepth: 2, gpuUtilization: 0.82 },
    lastHeartbeatTime: now - 20,
    lastDequeueTime: now - 120,
    lastBusyTime: now - 60,
    lastIdleTime: now - 600,
    currentRolloutId: 'ro-story-001',
    currentAttemptId: 'at-story-010',
  },
  {
    workerId: 'worker-west',
    status: 'busy',
    heartbeatStats: { queueDepth: 1 },
    lastHeartbeatTime: now - 45,
    lastDequeueTime: now - 300,
    lastBusyTime: now - 120,
    lastIdleTime: now - 4800,
    currentRolloutId: 'ro-story-003',
    currentAttemptId: 'at-story-033',
  },
  {
    workerId: 'worker-north',
    status: 'idle',
    heartbeatStats: { queueDepth: 0 },
    lastHeartbeatTime: now - 120,
    lastDequeueTime: now - 3600,
    lastBusyTime: now - 5400,
    lastIdleTime: now - 180,
    currentRolloutId: null,
    currentAttemptId: null,
  },
  {
    workerId: 'worker-south',
    status: 'idle',
    heartbeatStats: null,
    lastHeartbeatTime: now - 900,
    lastDequeueTime: now - 7200,
    lastBusyTime: now - 8600,
    lastIdleTime: now - 8600,
    currentRolloutId: null,
    currentAttemptId: null,
  },
  {
    workerId: 'worker-central',
    status: 'busy',
    heartbeatStats: { queueDepth: 3, cpuUtilization: 0.55 },
    lastHeartbeatTime: now - 8,
    lastDequeueTime: now - 45,
    lastBusyTime: now - 10,
    lastIdleTime: now - 900,
    currentRolloutId: 'ro-story-005',
    currentAttemptId: 'at-story-013',
  },
  {
    workerId: 'worker-standby',
    status: 'idle',
    heartbeatStats: { queueDepth: 0, threads: 32 },
    lastHeartbeatTime: now - 300,
    lastDequeueTime: now - 10800,
    lastBusyTime: now - 14400,
    lastIdleTime: now - 200,
    currentRolloutId: null,
    currentAttemptId: null,
  },
  {
    workerId: 'worker-observer',
    status: 'unknown',
    heartbeatStats: { queueDepth: 0 },
    lastHeartbeatTime: now - 30,
    lastDequeueTime: now - 6400,
    lastBusyTime: null,
    lastIdleTime: null,
    currentRolloutId: null,
    currentAttemptId: null,
  },
];

const defaultHandlers = createWorkersHandlers(sampleWorkers);

function createStoryStore(configOverrides?: Partial<typeof initialConfigState>) {
  return createAppStore({
    config: {
      ...initialConfigState,
      baseUrl: STORY_BASE_URL,
      autoRefreshMs: 0,
      ...configOverrides,
    },
    workers: initialWorkersUiState,
  });
}

function renderWithStore(configOverrides?: Partial<typeof initialConfigState>) {
  const store = createStoryStore(configOverrides);

  return (
    <Provider store={store}>
      <>
        <WorkersPage />
        <AppAlertBanner />
        <AppDrawerContainer />
      </>
    </Provider>
  );
}

function renderWithinAppLayout(configOverrides?: Partial<typeof initialConfigState>) {
  const store = createStoryStore(configOverrides);
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
            path: '/runners',
            element: <WorkersPage />,
          },
        ],
      },
    ],
    { initialEntries: ['/runners'] },
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

const manyWorkers = Array.from({ length: 80 }, (_, index) => {
  const suffix = (index + 1).toString().padStart(3, '0');
  const busy = index % 2 === 0;
  return {
    workerId: `worker-batch-${suffix}`,
    status: busy ? 'busy' : 'idle',
    heartbeatStats: busy ? { queueDepth: (index % 5) + 1 } : { queueDepth: 0 },
    lastHeartbeatTime: now - (index * 5 + 15),
    lastDequeueTime: now - (index * 20 + 60),
    lastBusyTime: busy ? now - (index * 10 + 30) : null,
    lastIdleTime: busy ? null : now - (index * 10 + 45),
    currentRolloutId: busy ? `ro-many-${suffix}` : null,
    currentAttemptId: busy ? `at-many-${suffix}` : null,
  } satisfies Worker;
});

export const Default: Story = {
  render: () => renderWithinAppLayout(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
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
    await canvas.findByText('worker-east');

    const searchInput = canvas.getByPlaceholderText('Search by Runner ID');
    await userEvent.type(searchInput, 'worker-west');

    await waitFor(() => {
      if (canvas.queryByText('worker-east')) {
        throw new Error('Expected filtered table to hide worker-east');
      }
      if (!canvas.queryByText('worker-west')) {
        throw new Error('Expected worker-west to remain visible');
      }
    });
  },
};

export const DrawerOpen: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('worker-east');

    const detailsButtons = await canvas.findAllByRole('button', { name: /detail/i });
    await userEvent.click(detailsButtons[0]);

    const body = within(document.body);
    await waitFor(() => {
      if (!body.queryByTestId('json-editor-container')) {
        throw new Error('Expected worker detail drawer with JSON view');
      }
    });
  },
};

export const ManyWorkers: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createWorkersHandlers(manyWorkers),
    },
  },
};

export const DarkTheme: Story = {
  render: () => renderWithStore({ theme: 'dark' }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: defaultHandlers,
    },
  },
};
