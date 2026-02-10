// Copyright (c) Microsoft. All rights reserved.

import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['scripts/headless-runner.ts'],
  format: ['cjs'],
  outDir: 'dist',
  clean: true,
  // Resolve @/ path aliases from tsconfig
  esbuildOptions(options) {
    options.alias = {
      '@': './src',
    };
  },
  // Bundle local src/ files but keep node_modules external
  external: [/node_modules/],
  // Target Node.js
  platform: 'node',
  target: 'node20',
});
