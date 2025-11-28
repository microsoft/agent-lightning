// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect } from 'react';
import { IconSearch } from '@tabler/icons-react';
import type { DataTableSortStatus } from 'mantine-datatable';
import { Skeleton, Stack, TextInput, Title } from '@mantine/core';
import { ResourcesTable, type ResourcesTableRecord } from '@/components/ResourcesTable.component';
import { ResourcesTree } from '@/components/ResourcesTree.component';
import { selectAutoRefreshMs } from '@/features/config';
import {
  resetResourcesFilters,
  selectResourcesPage,
  selectResourcesQueryArgs,
  selectResourcesRecordsPerPage,
  selectResourcesSearchTerm,
  selectResourcesSort,
  setResourcesPage,
  setResourcesRecordsPerPage,
  setResourcesSearchTerm,
  setResourcesSort,
} from '@/features/resources';
import { useGetResourcesQuery } from '@/features/rollouts';
import { hideAlert, showAlert } from '@/features/ui/alert';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import type { PaginatedResponse, Resources } from '@/types';
import { getErrorDescriptor } from '@/utils/error';

export function ResourcesPage() {
  const dispatch = useAppDispatch();
  const autoRefreshMs = useAppSelector(selectAutoRefreshMs);
  const searchTerm = useAppSelector(selectResourcesSearchTerm);
  const page = useAppSelector(selectResourcesPage);
  const recordsPerPage = useAppSelector(selectResourcesRecordsPerPage);
  const sort = useAppSelector(selectResourcesSort);
  const queryArgs = useAppSelector(selectResourcesQueryArgs);

  const resourcesQueryResult = useGetResourcesQuery(queryArgs, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  const resourcesData = resourcesQueryResult.data as PaginatedResponse<Resources> | undefined;
  const { isLoading, isFetching, isError, error, refetch } = resourcesQueryResult;

  const handleSearchTermChange = useCallback(
    (value: string) => {
      dispatch(setResourcesSearchTerm(value));
    },
    [dispatch],
  );

  const handleSortStatusChange = useCallback(
    (status: DataTableSortStatus<ResourcesTableRecord>) => {
      dispatch(
        setResourcesSort({
          column: status.columnAccessor,
          direction: status.direction,
        }),
      );
    },
    [dispatch],
  );

  const handlePageChange = useCallback(
    (nextPage: number) => {
      dispatch(setResourcesPage(nextPage));
    },
    [dispatch],
  );

  const handleRecordsPerPageChange = useCallback(
    (value: number) => {
      dispatch(setResourcesRecordsPerPage(value));
    },
    [dispatch],
  );

  const handleResetFilters = useCallback(() => {
    dispatch(resetResourcesFilters());
  }, [dispatch]);

  const showSkeleton = isLoading && !((resourcesData?.items?.length ?? 0) > 0);

  useEffect(() => {
    const descriptor = isError ? getErrorDescriptor(error) : null;
    if (isError) {
      const detailSuffix = descriptor ? ` (${descriptor})` : '';
      dispatch(
        showAlert({
          id: 'resources-fetch',
          message: `Unable to refresh resources${detailSuffix}. The table may be out of date until the connection recovers.`,
          tone: 'error',
        }),
      );
      return;
    }

    if (!isLoading && !isFetching) {
      dispatch(hideAlert({ id: 'resources-fetch' }));
    }
  }, [dispatch, error, isError, isFetching, isLoading]);

  useEffect(
    () => () => {
      dispatch(hideAlert({ id: 'resources-fetch' }));
    },
    [dispatch],
  );

  return (
    <Stack gap='md'>
      <Title order={1}>Resources</Title>

      <TextInput
        placeholder='Search by Resources ID'
        value={searchTerm}
        onChange={(event) => handleSearchTermChange(event.currentTarget.value)}
        leftSection={<IconSearch size={16} />}
        data-testid='resources-search-input'
        w='100%'
        style={{ maxWidth: 360 }}
      />

      {showSkeleton ? (
        <Skeleton height={360} radius='md' />
      ) : (
        <ResourcesTable
          resourcesList={resourcesData?.items}
          totalRecords={resourcesData?.total ?? 0}
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
          renderRowExpansion={({ resources }) => <ResourcesTree resources={resources} />}
        />
      )}
    </Stack>
  );
}
