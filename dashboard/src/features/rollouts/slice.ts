// Copyright (c) Microsoft. All rights reserved.

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { RolloutMode, RolloutStatus } from '../../types';

export type SortDirection = 'asc' | 'desc';

export type RolloutsSortState = {
  column: string;
  direction: SortDirection;
};

export type RolloutsUiState = {
  searchTerm: string;
  statusFilters: RolloutStatus[];
  modeFilters: RolloutMode[];
  page: number;
  recordsPerPage: number;
  sort: RolloutsSortState;
};

export const initialRolloutsUiState: RolloutsUiState = {
  searchTerm: '',
  statusFilters: [],
  modeFilters: [],
  page: 1,
  recordsPerPage: 100,
  sort: {
    column: 'startTimestamp',
    direction: 'desc',
  },
};

const rolloutsSlice = createSlice({
  name: 'rollouts',
  initialState: initialRolloutsUiState,
  reducers: {
    setRolloutsSearchTerm(state, action: PayloadAction<string>) {
      state.searchTerm = action.payload;
      state.page = 1;
    },
    setRolloutsStatusFilters(state, action: PayloadAction<RolloutStatus[]>) {
      state.statusFilters = action.payload;
      state.page = 1;
    },
    setRolloutsModeFilters(state, action: PayloadAction<RolloutMode[]>) {
      state.modeFilters = action.payload;
      state.page = 1;
    },
    setRolloutsPage(state, action: PayloadAction<number>) {
      state.page = action.payload;
    },
    setRolloutsRecordsPerPage(state, action: PayloadAction<number>) {
      state.recordsPerPage = action.payload;
      state.page = 1;
    },
    setRolloutsSort(state, action: PayloadAction<RolloutsSortState>) {
      state.sort = action.payload;
    },
    resetRolloutsFilters(state) {
      state.statusFilters = initialRolloutsUiState.statusFilters;
      state.modeFilters = initialRolloutsUiState.modeFilters;
      state.searchTerm = initialRolloutsUiState.searchTerm;
      state.page = initialRolloutsUiState.page;
      state.sort = initialRolloutsUiState.sort;
    },
  },
});

export const {
  setRolloutsSearchTerm,
  setRolloutsStatusFilters,
  setRolloutsModeFilters,
  setRolloutsPage,
  setRolloutsRecordsPerPage,
  setRolloutsSort,
  resetRolloutsFilters,
} = rolloutsSlice.actions;

export const rolloutsReducer = rolloutsSlice.reducer;
