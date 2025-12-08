// Copyright (c) Microsoft. All rights reserved.

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type SortDirection = 'asc' | 'desc';

export type TracesSortState = {
  column: string;
  direction: SortDirection;
};

export type TracesViewMode = 'table' | 'waterfall' | 'tree';

export type TracesUiState = {
  rolloutId: string | null;
  attemptId: string | null;
  searchTerm: string;
  page: number;
  recordsPerPage: number;
  sort: TracesSortState;
  viewMode: TracesViewMode;
};

export const initialTracesUiState: TracesUiState = {
  rolloutId: null,
  attemptId: null,
  searchTerm: '',
  page: 1,
  recordsPerPage: 100,
  sort: {
    column: 'startTime',
    direction: 'desc',
  },
  viewMode: 'table',
};

const tracesSlice = createSlice({
  name: 'traces',
  initialState: initialTracesUiState,
  reducers: {
    setTracesRolloutId(state, action: PayloadAction<string | null>) {
      state.rolloutId = action.payload;
      state.page = 1;
      state.attemptId = null;
    },
    setTracesAttemptId(state, action: PayloadAction<string | null>) {
      state.attemptId = action.payload;
      state.page = 1;
    },
    setTracesSearchTerm(state, action: PayloadAction<string>) {
      state.searchTerm = action.payload;
      state.page = 1;
    },
    setTracesPage(state, action: PayloadAction<number>) {
      state.page = action.payload;
    },
    setTracesRecordsPerPage(state, action: PayloadAction<number>) {
      state.recordsPerPage = action.payload;
      state.page = 1;
    },
    setTracesSort(state, action: PayloadAction<TracesSortState>) {
      state.sort = action.payload;
    },
    setTracesViewMode(state, action: PayloadAction<TracesViewMode>) {
      state.viewMode = action.payload;
    },
    resetTracesFilters(state) {
      state.searchTerm = initialTracesUiState.searchTerm;
      state.page = initialTracesUiState.page;
      state.sort = initialTracesUiState.sort;
    },
    hydrateTracesStateFromQuery(
      state,
      action: PayloadAction<{ rolloutId?: string | null; attemptId?: string | null }>,
    ) {
      const payload = action.payload;
      if (Object.hasOwn(payload, 'rolloutId')) {
        const nextRolloutId = payload.rolloutId ?? null;
        if (state.rolloutId !== nextRolloutId) {
          state.rolloutId = nextRolloutId;
          state.page = 1;
          state.attemptId = null;
        } else if (nextRolloutId === null) {
          state.attemptId = null;
        }
      }

      if (Object.hasOwn(payload, 'attemptId')) {
        if (state.rolloutId === null) {
          state.attemptId = null;
          return;
        }
        const nextAttemptId = payload.attemptId ?? null;
        if (state.attemptId !== nextAttemptId) {
          state.attemptId = nextAttemptId;
          state.page = 1;
        }
      }
    },
  },
});

export const {
  setTracesRolloutId,
  setTracesAttemptId,
  setTracesSearchTerm,
  setTracesPage,
  setTracesRecordsPerPage,
  setTracesSort,
  setTracesViewMode,
  resetTracesFilters,
  hydrateTracesStateFromQuery,
} = tracesSlice.actions;

export const tracesReducer = tracesSlice.reducer;
