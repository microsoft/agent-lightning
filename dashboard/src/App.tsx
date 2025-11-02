// Copyright (c) Microsoft. All rights reserved.

import '@mantine/core/styles.css';
import 'mantine-datatable/styles.css';
import './styles/theme.css';
import './styles/app.css';

import { MantineProvider } from '@mantine/core';
import { useColorScheme } from '@mantine/hooks';
import { shadcnCssVariableResolver } from './cssVariableResolver';
import { selectThemePreference } from './features/config/selectors';
import { Router } from './Router';
import { useAppSelector } from './store/hooks';
import { shadcnTheme } from './theme';

export default function App() {
  const themePreference = useAppSelector(selectThemePreference);
  const systemColorScheme = useColorScheme();
  const resolvedColorScheme = themePreference === 'system' ? systemColorScheme : themePreference;

  return (
    <MantineProvider
      theme={shadcnTheme}
      cssVariablesResolver={shadcnCssVariableResolver}
      forceColorScheme={resolvedColorScheme}
    >
      <Router />
    </MantineProvider>
  );
}
