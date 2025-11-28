// Copyright (c) Microsoft. All rights reserved.

import { createSelector } from '@reduxjs/toolkit';
import type { GetWorkersQueryArgs } from '@/features/rollouts';
import type { RootState } from '@/store';
import type { WorkersSortState } from './slice';

const WORKERS_SORT_FIELD_MAP: Record<string, string> = {
  workerId: 'worker_id',
  status: 'status',
  currentRolloutId: 'current_rollout_id',
  currentAttemptId: 'current_attempt_id',
  lastHeartbeatTime: 'last_heartbeat_time',
  lastDequeueTime: 'last_dequeue_time',
  lastBusyTime: 'last_busy_time',
  lastIdleTime: 'last_idle_time',
};

const resolveWorkersSortField = (sort: WorkersSortState): string =>
  WORKERS_SORT_FIELD_MAP[sort.column] ?? 'last_heartbeat_time';

export const selectWorkersUiState = (state: RootState) => state.workers;

export const selectWorkersSearchTerm = (state: RootState) => selectWorkersUiState(state).searchTerm;
export const selectWorkersPage = (state: RootState) => selectWorkersUiState(state).page;
export const selectWorkersRecordsPerPage = (state: RootState) => selectWorkersUiState(state).recordsPerPage;
export const selectWorkersSort = (state: RootState) => selectWorkersUiState(state).sort;

export const selectWorkersQueryArgs = createSelector(
  [selectWorkersSearchTerm, selectWorkersPage, selectWorkersRecordsPerPage, selectWorkersSort],
  (searchTerm, page, recordsPerPage, sort): GetWorkersQueryArgs => {
    const normalizedSearch = searchTerm.trim();
    const limit = Math.max(1, recordsPerPage);
    const offset = Math.max(0, (page - 1) * limit);
    const sortBy = resolveWorkersSortField(sort);

    return {
      limit,
      offset,
      sortBy,
      sortOrder: sort.direction,
      workerIdContains: normalizedSearch.length > 0 ? normalizedSearch : undefined,
    };
  },
);
