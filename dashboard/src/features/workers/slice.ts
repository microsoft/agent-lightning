// Copyright (c) Microsoft. All rights reserved.

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type SortDirection = 'asc' | 'desc';

export type WorkersSortState = {
  column: string;
  direction: SortDirection;
};

export type WorkersUiState = {
  searchTerm: string;
  page: number;
  recordsPerPage: number;
  sort: WorkersSortState;
};

export const initialWorkersUiState: WorkersUiState = {
  searchTerm: '',
  page: 1,
  recordsPerPage: 50,
  sort: {
    column: 'lastHeartbeatTime',
    direction: 'desc',
  },
};

const workersSlice = createSlice({
  name: 'workers',
  initialState: initialWorkersUiState,
  reducers: {
    setWorkersSearchTerm(state, action: PayloadAction<string>) {
      state.searchTerm = action.payload;
      state.page = 1;
    },
    setWorkersPage(state, action: PayloadAction<number>) {
      state.page = action.payload;
    },
    setWorkersRecordsPerPage(state, action: PayloadAction<number>) {
      state.recordsPerPage = action.payload;
      state.page = 1;
    },
    setWorkersSort(state, action: PayloadAction<WorkersSortState>) {
      state.sort = action.payload;
    },
    resetWorkersFilters(state) {
      state.searchTerm = initialWorkersUiState.searchTerm;
      state.page = initialWorkersUiState.page;
      state.recordsPerPage = initialWorkersUiState.recordsPerPage;
      state.sort = initialWorkersUiState.sort;
    },
  },
});

export const { setWorkersSearchTerm, setWorkersPage, setWorkersRecordsPerPage, setWorkersSort, resetWorkersFilters } =
  workersSlice.actions;

export const workersReducer = workersSlice.reducer;
