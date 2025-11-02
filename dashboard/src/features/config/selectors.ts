// Copyright (c) Microsoft. All rights reserved.

import type { RootState } from '../../store';

export const selectConfig = (state: RootState) => state.config;
export const selectAutoRefreshMs = (state: RootState) => state.config.autoRefreshMs;
export const selectBaseUrl = (state: RootState) => state.config.baseUrl;
export const selectThemePreference = (state: RootState) => state.config.theme;
