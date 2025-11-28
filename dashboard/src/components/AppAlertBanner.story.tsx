// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { initialConfigState } from '@/features/config/slice';
import { initialRolloutsUiState } from '@/features/rollouts/slice';
import type { AlertsState, AlertTone } from '@/features/ui/alert';
import { initialDrawerState } from '@/features/ui/drawer/slice';
import { createAppStore } from '@/store';
import { STORY_BASE_URL, STORY_DATE_NOW_MS } from '../../.storybook/constants';
import { AppAlertBanner } from './AppAlertBanner';

const meta: Meta<typeof AppAlertBanner> = {
  title: 'Components/AppAlertBanner',
  component: AppAlertBanner,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof AppAlertBanner>;

function renderWithAlert(message: string, tone: AlertTone) {
  const alertState: AlertsState = {
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

  const store = createAppStore({
    config: {
      ...initialConfigState,
      baseUrl: STORY_BASE_URL,
    },
    drawer: initialDrawerState,
    rollouts: initialRolloutsUiState,
    alert: alertState,
  });

  return (
    <Provider store={store}>
      <div style={{ padding: 24 }}>
        <AppAlertBanner />
      </div>
    </Provider>
  );
}

export const InfoAlert: Story = {
  render: () => renderWithAlert('Background synchronization completed successfully.', 'info'),
};

export const WarningAlert: Story = {
  render: () =>
    renderWithAlert('Rollout data may be stale. Check your network connection before continuing.', 'warning'),
};

export const ErrorAlert: Story = {
  render: () =>
    renderWithAlert('Unable to reach the Agent-lightning API. Retry or adjust the backend settings.', 'error'),
};

export const NoAlert: Story = {
  render: () => {
    const store = createAppStore({
      config: {
        ...initialConfigState,
        baseUrl: STORY_BASE_URL,
      },
      drawer: initialDrawerState,
      rollouts: initialRolloutsUiState,
      alert: { alerts: [] },
    });

    return (
      <Provider store={store}>
        <div style={{ padding: 24 }}>
          <AppAlertBanner />
        </div>
      </Provider>
    );
  },
};
