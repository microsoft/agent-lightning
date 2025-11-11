// Copyright (c) Microsoft. All rights reserved.

/// <reference types="vitest/config" />

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import react from '@vitejs/plugin-react';
import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

const dirname = typeof __dirname !== 'undefined' ? __dirname : path.dirname(fileURLToPath(import.meta.url));

// More info at: https://storybook.js.org/docs/next/writing-tests/integrations/vitest-addon
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = __dirname;
const appRoot = path.resolve(projectRoot, 'public');
export default defineConfig({
  root: appRoot,
  plugins: [react(), tsconfigPaths()],
  publicDir: path.resolve(projectRoot, 'static'),
  server: {
    fs: {
      allow: [projectRoot],
    },
  },
  build: {
    outDir: path.resolve(projectRoot, '../agentlightning/dashboard'),
    emptyOutDir: true,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.mjs',
    root: projectRoot,
    projects: [
      {
        extends: true,
        test: {
          name: 'unit',
          globalSetup: './vitest.global-setup.mjs',
        },
      },
      {
        // TODO: vitest for storybook has been setup but it's not working yet.
        extends: true,
        plugins: [
          // The plugin will run tests for the stories defined in your Storybook config
          // See options at: https://storybook.js.org/docs/next/writing-tests/integrations/vitest-addon#storybooktest
          storybookTest({
            configDir: path.join(dirname, '.storybook'),
          }),
        ],
        test: {
          name: 'storybook',
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({}),
            instances: [
              {
                browser: 'chromium',
              },
            ],
          },
          setupFiles: ['.storybook/vitest.setup.ts'],
        },
      },
    ],
  },
});
