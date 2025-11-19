// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { initialConfigState } from '@/features/config/slice';
import { createAppStore } from '@/store';
import type { ConfigState } from '@/types';
import { STORY_BASE_URL } from '../../.storybook/constants';
import { SettingsPage } from './Settings.page';

const meta: Meta<typeof SettingsPage> = {
  title: 'Pages/SettingsPage',
  component: SettingsPage,
  parameters: {
    layout: 'centered',
  },
};

export default meta;

type Story = StoryObj<typeof SettingsPage>;

function renderWithConfig(partial?: Partial<ConfigState>) {
  const store = createAppStore({
    config: {
      ...initialConfigState,
      baseUrl: STORY_BASE_URL,
      ...partial,
    },
  });

  return (
    <Provider store={store}>
      <SettingsPage />
    </Provider>
  );
}

export const Default: Story = {
  render: () => renderWithConfig(),
};

export const AutoRefreshEveryMinute: Story = {
  render: () => renderWithConfig({ autoRefreshMs: 60_000 }),
};

export const DarkThemeSelected: Story = {
  render: () => renderWithConfig({ theme: 'dark' }),
  parameters: {
    theme: 'dark',
  },
};

export const CustomBackendUrl: Story = {
  render: () => renderWithConfig({ baseUrl: 'https://api.agent-lightning.dev' }),
};
