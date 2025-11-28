// Copyright (c) Microsoft. All rights reserved.

import { createSelector } from '@reduxjs/toolkit';
import type { GetResourcesQueryArgs } from '@/features/rollouts';
import type { RootState } from '@/store';
import type { ResourcesSortState } from './slice';

const RESOURCES_SORT_FIELD_MAP: Record<string, string> = {
  resourcesId: 'resources_id',
  version: 'version',
  createTime: 'create_time',
  updateTime: 'update_time',
};

const resolveResourcesSortField = (sort: ResourcesSortState): string =>
  RESOURCES_SORT_FIELD_MAP[sort.column] ?? 'update_time';

export const selectResourcesUiState = (state: RootState) => state.resources;

export const selectResourcesSearchTerm = (state: RootState) => selectResourcesUiState(state).searchTerm;
export const selectResourcesPage = (state: RootState) => selectResourcesUiState(state).page;
export const selectResourcesRecordsPerPage = (state: RootState) => selectResourcesUiState(state).recordsPerPage;
export const selectResourcesSort = (state: RootState) => selectResourcesUiState(state).sort;

export const selectResourcesQueryArgs = createSelector(
  [selectResourcesSearchTerm, selectResourcesPage, selectResourcesRecordsPerPage, selectResourcesSort],
  (searchTerm, page, recordsPerPage, sort): GetResourcesQueryArgs => {
    const normalizedSearch = searchTerm.trim();
    const limit = Math.max(1, recordsPerPage);
    const offset = Math.max(0, (page - 1) * limit);
    const sortBy = resolveResourcesSortField(sort);

    return {
      limit,
      offset,
      sortBy,
      sortOrder: sort.direction,
      resourcesIdContains: normalizedSearch.length > 0 ? normalizedSearch : undefined,
    };
  },
);
