import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { configReducer } from '../features/config';
import { drawerReducer } from '../features/ui/drawer';
import { alertReducer } from '../features/ui/alert';
import { rolloutsReducer, rolloutsApi } from '../features/rollouts';
import { resourcesReducer } from '../features/resources';
import { tracesReducer } from '../features/traces';

const rootReducer = combineReducers({
  config: configReducer,
  drawer: drawerReducer,
  alert: alertReducer,
  rollouts: rolloutsReducer,
  resources: resourcesReducer,
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
