// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect } from 'react';
import { IconSearch } from '@tabler/icons-react';
import type { DataTableSortStatus } from 'mantine-datatable';
import { Skeleton, Stack, TextInput, Title } from '@mantine/core';
import { WorkersTable, type WorkersTableRecord } from '@/components/WorkersTable.component';
import { selectAutoRefreshMs } from '@/features/config';
import { hideAlert, showAlert } from '@/features/ui/alert';
import { openDrawer } from '@/features/ui/drawer';
import {
  resetWorkersFilters,
  selectWorkersPage,
  selectWorkersQueryArgs,
  selectWorkersRecordsPerPage,
  selectWorkersSearchTerm,
  selectWorkersSort,
  setWorkersPage,
  setWorkersRecordsPerPage,
  setWorkersSearchTerm,
  setWorkersSort,
  useGetWorkersQuery,
} from '@/features/workers';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import type { PaginatedResponse, Worker } from '@/types';
import { getErrorDescriptor } from '@/utils/error';

export function WorkersPage() {
  const dispatch = useAppDispatch();
  const autoRefreshMs = useAppSelector(selectAutoRefreshMs);
  const searchTerm = useAppSelector(selectWorkersSearchTerm);
  const page = useAppSelector(selectWorkersPage);
  const recordsPerPage = useAppSelector(selectWorkersRecordsPerPage);
  const sort = useAppSelector(selectWorkersSort);
  const queryArgs = useAppSelector(selectWorkersQueryArgs);

  const workersQueryResult = useGetWorkersQuery(queryArgs, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  const workersData = workersQueryResult.data as PaginatedResponse<Worker> | undefined;
  const { isLoading, isFetching, isError, error, refetch } = workersQueryResult;

  const handleSearchTermChange = useCallback(
    (value: string) => {
      dispatch(setWorkersSearchTerm(value));
    },
    [dispatch],
  );

  const handleSortStatusChange = useCallback(
    (status: DataTableSortStatus<WorkersTableRecord>) => {
      dispatch(
        setWorkersSort({
          column: status.columnAccessor,
          direction: status.direction,
        }),
      );
    },
    [dispatch],
  );

  const handlePageChange = useCallback(
    (nextPage: number) => {
      dispatch(setWorkersPage(nextPage));
    },
    [dispatch],
  );

  const handleRecordsPerPageChange = useCallback(
    (value: number) => {
      dispatch(setWorkersRecordsPerPage(value));
    },
    [dispatch],
  );

  const handleResetFilters = useCallback(() => {
    dispatch(resetWorkersFilters());
  }, [dispatch]);

  const handleShowWorkerDetails = useCallback(
    (worker: Worker) => {
      dispatch(
        openDrawer({
          type: 'worker-detail',
          worker,
        }),
      );
    },
    [dispatch],
  );

  const hasWorkers = Array.isArray(workersData?.items) && workersData.items.length > 0;
  const showSkeleton = isLoading && !hasWorkers;

  useEffect(() => {
    if (isError) {
      const descriptor = getErrorDescriptor(error);
      const suffix = descriptor ? ` (${descriptor})` : '';
      dispatch(
        showAlert({
          id: 'workers-fetch',
          message: `Unable to refresh workers${suffix}. The table may be out of date until the connection recovers.`,
          tone: 'error',
        }),
      );
      return;
    }

    if (!isLoading && !isFetching) {
      dispatch(hideAlert({ id: 'workers-fetch' }));
    }
  }, [dispatch, error, isError, isFetching, isLoading]);

  useEffect(
    () => () => {
      dispatch(hideAlert({ id: 'workers-fetch' }));
    },
    [dispatch],
  );

  return (
    <Stack gap='md'>
      <Title order={1}>Runners</Title>

      <TextInput
        placeholder='Search by Runner ID'
        value={searchTerm}
        onChange={(event) => handleSearchTermChange(event.currentTarget.value)}
        leftSection={<IconSearch size={16} />}
        data-testid='workers-search-input'
        w='100%'
        style={{ maxWidth: 360 }}
      />

      {showSkeleton ? (
        <Skeleton height={360} radius='md' />
      ) : (
        <WorkersTable
          workers={workersData?.items}
          totalRecords={workersData?.total ?? 0}
          isFetching={isFetching}
          isError={isError}
          error={error}
          searchTerm={searchTerm}
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onSortStatusChange={handleSortStatusChange}
          onPageChange={handlePageChange}
          onRecordsPerPageChange={handleRecordsPerPageChange}
          onResetFilters={handleResetFilters}
          onRefetch={refetch}
          onShowDetails={handleShowWorkerDetails}
        />
      )}
    </Stack>
  );
}
