// Copyright (c) Microsoft. All rights reserved.

import '@mantine/core/styles.css';
import 'mantine-datatable/styles.css';
import '../src/styles/theme.css';
import '../src/styles/app.css';

import { initialize, mswLoader } from 'msw-storybook-addon';
import { ColorSchemeScript, MantineProvider } from '@mantine/core';
import { shadcnCssVariableResolver } from '../src/cssVariableResolver';
import { theme as mantineTheme } from '../src/theme';
import { STORY_DATE_NOW_MS } from './constants';

type ColorSchemeValue = 'light' | 'dark';

initialize({
  onUnhandledRequest: 'bypass',
  serviceWorker: {
    url: '/mockServiceWorker.js',
  },
});

const fixedDateNow = (() => {
  const patched = Date.now as typeof Date.now & { __storybookPatched?: boolean };
  if (patched.__storybookPatched) {
    return patched;
  }
  const replacement = (() => STORY_DATE_NOW_MS) as typeof Date.now & { __storybookPatched?: boolean };
  replacement.__storybookPatched = true;
  return replacement;
})();

Date.now = fixedDateNow;

export const parameters = {
  layout: 'fullscreen',
  options: {
    showPanel: false,
    // @ts-expect-error – storybook throws build error for (a: any, b: any)
    storySort: (a, b) => a.title.localeCompare(b.title, undefined, { numeric: true }),
  },
  backgrounds: { disable: true },
  viewport: {
    options: {
      md: { name: 'md', styles: { width: '1280px', height: '800px' } },
      lg: { name: 'lg', styles: { width: '1920px', height: '1080px' } },
      xl: { name: 'xl', styles: { width: '2560px', height: '1440px' } },
    },
  },
};

export const globalTypes = {
  theme: {
    name: 'Theme',
    description: 'Mantine color scheme',
    defaultValue: 'light',
    toolbar: {
      icon: 'mirror',
      items: [
        { value: 'light', title: 'Light' },
        { value: 'dark', title: 'Dark' },
      ],
    },
  },
};

export const decorators = [
  (Story: any, context: any) => {
    const scheme = (context.parameters.theme ?? context.globals.theme ?? 'light') as ColorSchemeValue;
    return (
      <MantineProvider theme={mantineTheme} cssVariablesResolver={shadcnCssVariableResolver} forceColorScheme={scheme}>
        <ColorSchemeScript />
        <Story />
      </MantineProvider>
    );
  },
];

export const loaders = [mswLoader];
