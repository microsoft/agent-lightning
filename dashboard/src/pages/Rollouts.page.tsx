// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect } from 'react';
import { IconSearch } from '@tabler/icons-react';
import type { DataTableColumn, DataTableSortStatus } from 'mantine-datatable';
import { Skeleton, Stack, TextInput, Title } from '@mantine/core';
import { RolloutAttemptsTable, RolloutTable, type RolloutTableRecord } from '@/components/RolloutTable.component';
import { selectAutoRefreshMs } from '@/features/config';
import {
  resetRolloutsFilters,
  selectRolloutsModeFilters,
  selectRolloutsPage,
  selectRolloutsQueryArgs,
  selectRolloutsRecordsPerPage,
  selectRolloutsSearchTerm,
  selectRolloutsSort,
  selectRolloutsStatusFilters,
  setRolloutsModeFilters,
  setRolloutsPage,
  setRolloutsRecordsPerPage,
  setRolloutsSearchTerm,
  setRolloutsSort,
  setRolloutsStatusFilters,
  useGetRolloutAttemptsQuery,
  useGetRolloutsQuery,
  type Rollout,
  type RolloutMode,
  type RolloutStatus,
} from '@/features/rollouts';
import { hideAlert, showAlert } from '@/features/ui/alert';
import { openDrawer } from '@/features/ui/drawer';
import { useAppDispatch, useAppSelector } from '@/store/hooks';

function RolloutAttemptsContent({
  rollout,
  columns,
}: {
  rollout: Rollout;
  columns: DataTableColumn<RolloutTableRecord>[];
}) {
  const { data, isFetching, isError, refetch } = useGetRolloutAttemptsQuery({
    rolloutId: rollout.rolloutId,
    limit: 100,
    sortBy: 'sequence_id',
    sortOrder: 'desc',
  });

  return (
    <RolloutAttemptsTable
      rollout={rollout}
      attempts={data?.items}
      isFetching={isFetching}
      isError={isError}
      onRetry={refetch}
      columns={columns}
    />
  );
}

function toRolloutFromRecord(record: RolloutTableRecord): Rollout {
  const { rolloutId, input, startTime, endTime, mode, resourcesId, status, config, metadata, attempt } = record;

  return {
    rolloutId,
    input,
    startTime,
    endTime,
    mode,
    resourcesId,
    status,
    config,
    metadata,
    attempt: attempt ?? null,
  };
}

export function RolloutsPage() {
  const dispatch = useAppDispatch();
  const autoRefreshMs = useAppSelector(selectAutoRefreshMs);
  const searchTerm = useAppSelector(selectRolloutsSearchTerm);
  const statusFilters = useAppSelector(selectRolloutsStatusFilters);
  const modeFilters = useAppSelector(selectRolloutsModeFilters);
  const page = useAppSelector(selectRolloutsPage);
  const recordsPerPage = useAppSelector(selectRolloutsRecordsPerPage);
  const sort = useAppSelector(selectRolloutsSort);
  const rolloutsQueryArgs = useAppSelector(selectRolloutsQueryArgs);

  const {
    data: rolloutsData,
    isLoading,
    isFetching,
    isError,
    error,
    refetch,
  } = useGetRolloutsQuery(rolloutsQueryArgs, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  const handleSearchTermChange = useCallback(
    (value: string) => {
      dispatch(setRolloutsSearchTerm(value));
    },
    [dispatch],
  );

  const handleStatusFilterChange = useCallback(
    (values: RolloutStatus[]) => {
      dispatch(setRolloutsStatusFilters(values));
    },
    [dispatch],
  );

  const handleStatusFilterReset = useCallback(() => {
    dispatch(setRolloutsStatusFilters([]));
  }, [dispatch]);

  const handleModeFilterChange = useCallback(
    (values: RolloutMode[]) => {
      dispatch(setRolloutsModeFilters(values));
    },
    [dispatch],
  );

  const handleModeFilterReset = useCallback(() => {
    dispatch(setRolloutsModeFilters([]));
  }, [dispatch]);

  const handleSortStatusChange = useCallback(
    (status: DataTableSortStatus<RolloutTableRecord>) => {
      dispatch(
        setRolloutsSort({
          column: status.columnAccessor as string,
          direction: status.direction,
        }),
      );
    },
    [dispatch],
  );

  const handlePageChange = useCallback(
    (nextPage: number) => {
      dispatch(setRolloutsPage(nextPage));
    },
    [dispatch],
  );

  const handleRecordsPerPageChange = useCallback(
    (value: number) => {
      dispatch(setRolloutsRecordsPerPage(value));
    },
    [dispatch],
  );

  const handleResetFilters = useCallback(() => {
    dispatch(resetRolloutsFilters());
  }, [dispatch]);

  const handleViewRawJson = useCallback(
    (record: RolloutTableRecord) => {
      dispatch(
        openDrawer({
          type: 'rollout-json',
          rollout: toRolloutFromRecord(record),
          attempt: record.attempt ?? null,
          isNested: record.isNested,
        }),
      );
    },
    [dispatch],
  );

  const handleViewTraces = useCallback(
    (record: RolloutTableRecord) => {
      dispatch(
        openDrawer({
          type: 'rollout-traces',
          rollout: toRolloutFromRecord(record),
          attempt: record.attempt ?? null,
          isNested: record.isNested,
        }),
      );
    },
    [dispatch],
  );

  const hasRollouts = Array.isArray(rolloutsData?.items) && rolloutsData.items.length > 0;
  const showSkeleton = isLoading && !hasRollouts;

  useEffect(() => {
    if (isError) {
      dispatch(
        showAlert({
          id: 'rollouts-fetch',
          message: 'Unable to refresh rollouts. The list below may be out of date until the connection recovers.',
          tone: 'error',
        }),
      );
      return;
    }

    if (!isLoading && !isFetching) {
      dispatch(hideAlert({ id: 'rollouts-fetch' }));
    }
  }, [dispatch, isError, isFetching, isLoading]);

  useEffect(
    () => () => {
      dispatch(hideAlert({ id: 'rollouts-fetch' }));
    },
    [dispatch],
  );

  return (
    <Stack gap='md'>
      <Title order={1}>Rollouts</Title>

      <TextInput
        placeholder='Search by Rollout ID'
        value={searchTerm}
        onChange={(event) => handleSearchTermChange(event.currentTarget.value)}
        leftSection={<IconSearch size={16} />}
        data-testid='rollouts-search-input'
        w='100%'
        style={{ maxWidth: 360 }}
      />

      {showSkeleton ? (
        <Skeleton height={360} radius='md' />
      ) : (
        <RolloutTable
          rollouts={rolloutsData?.items}
          totalRecords={rolloutsData?.total ?? 0}
          isFetching={isFetching}
          isError={isError}
          error={error}
          searchTerm={searchTerm}
          statusFilters={statusFilters}
          modeFilters={modeFilters}
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onStatusFilterChange={handleStatusFilterChange}
          onStatusFilterReset={handleStatusFilterReset}
          onModeFilterChange={handleModeFilterChange}
          onModeFilterReset={handleModeFilterReset}
          onSortStatusChange={handleSortStatusChange}
          onPageChange={handlePageChange}
          onRecordsPerPageChange={handleRecordsPerPageChange}
          onResetFilters={handleResetFilters}
          onRefetch={refetch}
          onViewRawJson={handleViewRawJson}
          onViewTraces={handleViewTraces}
          renderRowExpansion={({ rollout, columns }) => <RolloutAttemptsContent rollout={rollout} columns={columns} />}
        />
      )}
    </Stack>
  );
}
