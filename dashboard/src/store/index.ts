// Copyright (c) Microsoft. All rights reserved.

import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { configReducer } from '../features/config';
import { resourcesReducer } from '../features/resources';
import { rolloutsApi, rolloutsReducer } from '../features/rollouts';
import { tracesReducer } from '../features/traces';
import { alertReducer } from '../features/ui/alert';
import { drawerReducer } from '../features/ui/drawer';
import { workersReducer } from '../features/workers';

const rootReducer = combineReducers({
  config: configReducer,
  drawer: drawerReducer,
  alert: alertReducer,
  rollouts: rolloutsReducer,
  resources: resourcesReducer,
  workers: workersReducer,
  traces: tracesReducer,
  [rolloutsApi.reducerPath]: rolloutsApi.reducer,
});

export type RootState = ReturnType<typeof rootReducer>;

export const createAppStore = (preloadedState?: Partial<RootState>) =>
  configureStore({
    reducer: rootReducer,
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(rolloutsApi.middleware),
    preloadedState,
  });

export const store = createAppStore();

export type AppStore = ReturnType<typeof createAppStore>;
export type AppDispatch = AppStore['dispatch'];
