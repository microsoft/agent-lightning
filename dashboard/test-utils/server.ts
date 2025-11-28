// Copyright (c) Microsoft. All rights reserved.

import { setBaseUrl } from '@/features/config';
import { createAppStore, type AppStore } from '@/store';

const resolvePythonServerBaseUrl = (): string => {
  const host = process.env.VITEST_PYTHON_HOST ?? '127.0.0.1';
  const port = process.env.VITEST_PYTHON_PORT ?? '8765';
  const configured = process.env.VITEST_PYTHON_URL;
  const base = configured && configured.trim().length > 0 ? configured : `http://${host}:${port}`;
  return base.replace(/\/$/, '');
};

export const getPythonServerBaseUrl = (): string => resolvePythonServerBaseUrl();

export const createServerBackedStore = (preloadedState?: Parameters<typeof createAppStore>[0]): AppStore => {
  const store = createAppStore(preloadedState);
  store.dispatch(setBaseUrl(getPythonServerBaseUrl()));
  return store;
};
