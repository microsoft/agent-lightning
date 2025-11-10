// Copyright (c) Microsoft. All rights reserved.

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

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
    outDir: path.resolve(projectRoot, 'dist'),
    emptyOutDir: true,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.mjs',
    globalSetup: './vitest.global-setup.mjs',
    root: projectRoot,
  },
});
