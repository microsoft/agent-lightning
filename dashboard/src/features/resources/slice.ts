// Copyright (c) Microsoft. All rights reserved.

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type SortDirection = 'asc' | 'desc';

export type ResourcesSortState = {
  column: string;
  direction: SortDirection;
};

export type ResourcesUiState = {
  searchTerm: string;
  page: number;
  recordsPerPage: number;
  sort: ResourcesSortState;
};

export const initialResourcesUiState: ResourcesUiState = {
  searchTerm: '',
  page: 1,
  recordsPerPage: 50,
  sort: {
    column: 'updateTime',
    direction: 'desc',
  },
};

const resourcesSlice = createSlice({
  name: 'resources',
  initialState: initialResourcesUiState,
  reducers: {
    setResourcesSearchTerm(state, action: PayloadAction<string>) {
      state.searchTerm = action.payload;
      state.page = 1;
    },
    setResourcesPage(state, action: PayloadAction<number>) {
      state.page = action.payload;
    },
    setResourcesRecordsPerPage(state, action: PayloadAction<number>) {
      state.recordsPerPage = action.payload;
      state.page = 1;
    },
    setResourcesSort(state, action: PayloadAction<ResourcesSortState>) {
      state.sort = action.payload;
    },
    resetResourcesFilters(state) {
      state.searchTerm = initialResourcesUiState.searchTerm;
      state.page = initialResourcesUiState.page;
      state.recordsPerPage = initialResourcesUiState.recordsPerPage;
      state.sort = initialResourcesUiState.sort;
    },
  },
});

export const {
  setResourcesSearchTerm,
  setResourcesPage,
  setResourcesRecordsPerPage,
  setResourcesSort,
  resetResourcesFilters,
} = resourcesSlice.actions;

export const resourcesReducer = resourcesSlice.reducer;
