// Copyright (c) Microsoft. All rights reserved.

import { useEffect, useState } from 'react';
import { Divider, Select, Stack, Text, TextInput, Title } from '@mantine/core';
import { selectConfig } from '../features/config';
import { setAutoRefreshMs, setBaseUrl, setTheme } from '../features/config/slice';
import { useAppDispatch, useAppSelector } from '../store/hooks';

const AUTO_REFRESH_OPTIONS = [
  { label: 'Off', value: '0' },
  // TODO: Support real auto-refresh
  { label: 'Every 5 seconds', value: '5000', disabled: true },
  { label: 'Every 15 seconds', value: '15000', disabled: true },
  { label: 'Every 60 seconds', value: '60000', disabled: true },
  { label: 'Every 5 minutes', value: '300000', disabled: true },
];

const THEME_OPTIONS = [
  { label: 'System default', value: 'system' },
  { label: 'Light', value: 'light' },
  { label: 'Dark', value: 'dark' },
];

export function SettingsPage() {
  const dispatch = useAppDispatch();
  const config = useAppSelector(selectConfig);
  const [baseUrlInput, setBaseUrlInput] = useState(config.baseUrl);

  useEffect(() => {
    setBaseUrlInput(config.baseUrl);
  }, [config.baseUrl]);

  return (
    <Stack gap='lg' p='md' data-testid='settings-page'>
      <Title order={1}>Settings</Title>
      <Stack gap='md'>
        <Stack gap={4}>
          <Text fw={600} size='sm'>
            Backend base URL
          </Text>
          <Text size='xs' c='dimmed'>
            The dashboard uses this address for health checks and API calls.
          </Text>
          <TextInput
            placeholder='http://localhost:8000'
            value={baseUrlInput}
            onChange={(event) => setBaseUrlInput(event.currentTarget.value)}
            onBlur={() => dispatch(setBaseUrl(baseUrlInput.trim()))}
            data-testid='settings-base-url'
          />
        </Stack>
        <Divider />
        <Stack gap={4}>
          <Text fw={600} size='sm'>
            Auto-refresh interval
          </Text>
          <Text size='xs' c='dimmed'>
            Controls how often the dashboard polls the server.
          </Text>
          <Select
            value={String(config.autoRefreshMs)}
            data={AUTO_REFRESH_OPTIONS}
            onChange={(value) => {
              if (!value) {
                dispatch(setAutoRefreshMs(0));
                return;
              }
              dispatch(setAutoRefreshMs(Number(value)));
            }}
            data-testid='settings-auto-refresh'
          />
        </Stack>
        <Divider />
        <Stack gap={4}>
          <Text fw={600} size='sm'>
            Theme
          </Text>
          <Text size='xs' c='dimmed'>
            Choose the color scheme for the dashboard.
          </Text>
          <Select
            value={config.theme}
            data={THEME_OPTIONS}
            onChange={(value) => {
              if (!value) {
                return;
              }
              dispatch(setTheme(value as typeof config.theme));
            }}
            data-testid='settings-theme'
          />
        </Stack>
      </Stack>
    </Stack>
  );
}
