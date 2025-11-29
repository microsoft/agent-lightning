// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { http, HttpResponse } from 'msw';
import { Provider } from 'react-redux';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { Stack, Text } from '@mantine/core';
import { initialConfigState } from '@/features/config/slice';
import type { AlertsState, AlertTone } from '@/features/ui/alert';
import { createAppStore } from '@/store';
import { STORY_BASE_URL, STORY_DATE_NOW_MS } from '../../.storybook/constants';
import { AppLayout, AppLayoutProps } from './AppLayout';

const Placeholder = ({ title, description }: { title: string; description: string }) => (
  <Stack gap='sm' p='lg'>
    <Text size='lg' fw={600}>
      {title}
    </Text>
    <Text size='sm' c='dimmed'>
      {description}
    </Text>
  </Stack>
);

const ROUTES = [
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        path: 'rollouts',
        element: <Placeholder title='Rollouts' description='Track the rollout queue and status.' />,
      },
      {
        path: 'resources',
        element: <Placeholder title='Resources' description='Inspect resource snapshots and metadata.' />,
      },
      {
        path: 'traces',
        element: <Placeholder title='Traces' description='Browse telemetry spans across attempts.' />,
      },
      {
        path: 'runners',
        element: <Placeholder title='Runners' description='Monitor runner activity and status.' />,
      },
      {
        path: 'settings',
        element: (
          <Placeholder title='Settings' description='Configure server connection, refresh cadence, and appearance.' />
        ),
      },
    ],
  },
];

function createAlertState(message: string, tone: AlertTone): AlertsState {
  return {
    alerts: [
      {
        id: 'storybook-alert',
        message,
        tone,
        isVisible: true,
        createdAt: STORY_DATE_NOW_MS,
      },
    ],
  };
}

function renderAppLayout(args: AppLayoutProps, initialEntry = '/rollouts', alertState?: AlertsState) {
  const store = createAppStore({
    config: {
      ...initialConfigState,
      baseUrl: STORY_BASE_URL,
    },
    alert: alertState ?? { alerts: [] },
  });
  const router = createMemoryRouter(
    ROUTES.map((route) => ({
      ...route,
      element: <AppLayout {...args} />,
    })),
    { initialEntries: [initialEntry] },
  );

  return (
    <Provider store={store}>
      <RouterProvider router={router} />
    </Provider>
  );
}

const meta: Meta<AppLayoutProps> = {
  title: 'Layouts/AppLayout',
  component: AppLayout,
  parameters: {
    layout: 'fullscreen',
  },
  render: (args) => renderAppLayout(args, '/rollouts'),
  args: {
    config: {
      baseUrl: STORY_BASE_URL,
      autoRefreshMs: 0,
    },
  },
};

export default meta;

type Story = StoryObj<AppLayoutProps>;

export const NoServerConfigured: Story = {
  args: {
    config: {
      baseUrl: '',
      autoRefreshMs: 0,
    },
  },
};

export const ServerOnline: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get(`${STORY_BASE_URL}/v1/agl/health`, () => HttpResponse.json({ status: 'ok' }, { status: 200 })),
      ],
    },
  },
};

export const ServerOffline: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get(`${STORY_BASE_URL}/v1/agl/health`, () =>
          HttpResponse.json({ message: 'unavailable' }, { status: 503 }),
        ),
      ],
    },
  },
};

export const ResourcesNavActive: Story = {
  render: (args) => renderAppLayout(args, '/resources'),
};

export const TracesNavActive: Story = {
  render: (args) => renderAppLayout(args, '/traces'),
};

export const SettingsNavActive: Story = {
  render: (args) => renderAppLayout(args, '/settings'),
};

export const PollingEveryFiveSeconds: Story = {
  args: {
    config: {
      baseUrl: STORY_BASE_URL,
      autoRefreshMs: 5000,
    },
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${STORY_BASE_URL}/v1/agl/health`, () => HttpResponse.json({ status: 'ok' }, { status: 200 })),
      ],
    },
  },
};

export const InfoAlertActive: Story = {
  render: (args) =>
    renderAppLayout(args, '/rollouts', createAlertState('Background synchronization completed successfully.', 'info')),
};

export const WarningAlertActive: Story = {
  render: (args) =>
    renderAppLayout(
      args,
      '/rollouts',
      createAlertState('Rollout data may be stale. Check connectivity before proceeding.', 'warning'),
    ),
};

export const ErrorAlertActive: Story = {
  render: (args) =>
    renderAppLayout(
      args,
      '/rollouts',
      createAlertState('Unable to reach Agent-lightning API. Retry or adjust server settings.', 'error'),
    ),
};
