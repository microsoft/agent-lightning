// Copyright (c) Microsoft. All rights reserved.

import { createSelector } from '@reduxjs/toolkit';
import type { GetSpansQueryArgs } from '@/features/rollouts';
import type { RootState } from '@/store';
import type { TracesSortState } from './slice';

export const selectTracesState = (state: RootState) => state.traces;

export const selectTracesRolloutId = (state: RootState) => selectTracesState(state).rolloutId;

export const selectTracesAttemptId = (state: RootState) => selectTracesState(state).attemptId;

export const selectTracesSearchTerm = (state: RootState) => selectTracesState(state).searchTerm;

export const selectTracesPage = (state: RootState) => selectTracesState(state).page;

export const selectTracesRecordsPerPage = (state: RootState) => selectTracesState(state).recordsPerPage;

export const selectTracesSort = (state: RootState) => selectTracesState(state).sort;

export const selectTracesViewMode = (state: RootState) => selectTracesState(state).viewMode;

const TRACES_SORT_FIELD_MAP: Record<string, string> = {
  name: 'name',
  traceId: 'trace_id',
  spanId: 'span_id',
  parentId: 'parent_id',
  statusCode: 'status_code',
  startTime: 'start_time',
  duration: 'duration',
};

const resolveTracesSortField = (sort: TracesSortState): string => TRACES_SORT_FIELD_MAP[sort.column] ?? 'start_time';

export const selectTracesQueryArgs = createSelector(
  [
    selectTracesRolloutId,
    selectTracesAttemptId,
    selectTracesSearchTerm,
    selectTracesPage,
    selectTracesRecordsPerPage,
    selectTracesSort,
  ],
  (rolloutId, attemptId, searchTerm, page, recordsPerPage, sort): GetSpansQueryArgs | undefined => {
    if (!rolloutId) {
      return undefined;
    }
    const limit = Math.max(1, recordsPerPage);
    const offset = Math.max(0, (page - 1) * limit);
    const normalizedSearch = searchTerm.trim();
    const containsValue = normalizedSearch.length > 0 ? normalizedSearch : undefined;

    const sortBy = resolveTracesSortField(sort);

    return {
      rolloutId,
      attemptId: attemptId ?? undefined,
      limit,
      offset,
      sortBy,
      sortOrder: sort.direction,
      traceIdContains: containsValue,
      spanIdContains: containsValue,
      nameContains: containsValue,
      filterLogic: containsValue ? 'or' : undefined,
    };
  },
);
