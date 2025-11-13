// Copyright (c) Microsoft. All rights reserved.

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { ConfigState, ThemePreference } from '@/types';

export const initialConfigState: ConfigState = {
  baseUrl: typeof window !== 'undefined' ? window.location.origin : '',
  autoRefreshMs: 0,
  theme: 'system',
};

const configSlice = createSlice({
  name: 'config',
  initialState: initialConfigState,
  reducers: {
    setBaseUrl(state, action: PayloadAction<string>) {
      state.baseUrl = action.payload;
    },
    setAutoRefreshMs(state, action: PayloadAction<number>) {
      state.autoRefreshMs = action.payload;
    },
    setTheme(state, action: PayloadAction<ThemePreference>) {
      state.theme = action.payload;
    },
  },
});

export const { setAutoRefreshMs, setBaseUrl, setTheme } = configSlice.actions;

export const configReducer = configSlice.reducer;
