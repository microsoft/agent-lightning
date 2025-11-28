// Copyright (c) Microsoft. All rights reserved.

import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from '@/store';
import type { RolloutMode, RolloutStatus } from '@/types';
import type { RolloutsSortState } from './slice';

const ROLLOUTS_SORT_FIELD_MAP: Record<string, string> = {
  rolloutId: 'rollout_id',
  attemptId: 'attempt_id',
  statusValue: 'status',
  resourcesId: 'resources_id',
  mode: 'mode',
  startTimestamp: 'start_time',
  durationSeconds: 'duration',
  lastHeartbeatTimestamp: 'last_heartbeat_time',
  workerId: 'worker_id',
};

const resolveRolloutsSortField = (sort: RolloutsSortState): string =>
  ROLLOUTS_SORT_FIELD_MAP[sort.column] ?? 'start_time';

export const selectRolloutsUiState = (state: RootState) => state.rollouts;
export const selectRolloutsSearchTerm = (state: RootState) => state.rollouts.searchTerm;
export const selectRolloutsStatusFilters = (state: RootState) => state.rollouts.statusFilters;
export const selectRolloutsModeFilters = (state: RootState) => state.rollouts.modeFilters;
export const selectRolloutsPage = (state: RootState) => state.rollouts.page;
export const selectRolloutsRecordsPerPage = (state: RootState) => state.rollouts.recordsPerPage;
export const selectRolloutsSort = (state: RootState) => state.rollouts.sort;

export const selectRolloutsQueryArgs = createSelector(
  [
    selectRolloutsSearchTerm,
    selectRolloutsStatusFilters,
    selectRolloutsModeFilters,
    selectRolloutsPage,
    selectRolloutsRecordsPerPage,
    selectRolloutsSort,
  ],
  (
    searchTerm: string,
    statusFilters: RolloutStatus[],
    modeFilters: RolloutMode[],
    page: number,
    recordsPerPage: number,
    sort: RolloutsSortState,
  ) => {
    const normalizedSearch = searchTerm.trim();
    const limit = Math.max(1, recordsPerPage);
    const offset = Math.max(0, (page - 1) * limit);
    const sortBy = resolveRolloutsSortField(sort);

    return {
      limit,
      offset,
      sortBy,
      sortOrder: sort.direction,
      statusIn: statusFilters.length > 0 ? statusFilters : undefined,
      rolloutIdContains: normalizedSearch.length > 0 ? normalizedSearch : undefined,
      modeIn: modeFilters.length > 0 ? modeFilters : undefined,
    };
  },
);
