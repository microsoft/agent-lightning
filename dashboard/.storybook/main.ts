// Copyright (c) Microsoft. All rights reserved.

import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  core: {
    disableWhatsNewNotifications: true,
    disableTelemetry: true,
    enableCrashReports: false,
  },
  stories: ['../src/**/*.mdx', '../src/**/*.story.@(js|jsx|ts|tsx)'],
  staticDirs: ['../static'],
  addons: ['@storybook/addon-themes', '@storybook/addon-vitest'],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
};

export default config;
