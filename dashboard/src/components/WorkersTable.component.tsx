// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect, useMemo } from 'react';
import { IconCheck, IconCopy, IconInfoCircle, IconRefresh } from '@tabler/icons-react';
import { DataTable, type DataTableColumn, type DataTableSortStatus } from 'mantine-datatable';
import { ActionIcon, Badge, Box, Button, CopyButton, Group, Stack, Text, Tooltip } from '@mantine/core';
import { useElementSize, useViewportSize } from '@mantine/hooks';
import { getLayoutAwareWidth } from '@/layouts/helper';
import type { Worker } from '@/types';
import { getErrorDescriptor } from '@/utils/error';
import { formatDateTime, formatRelativeTime, formatStatusLabel } from '@/utils/format';
import { createResponsiveColumns, type ColumnVisibilityConfig } from '@/utils/table';

const DEFAULT_RECORDS_PER_PAGE_OPTIONS = [50, 100, 200, 500];

const COLUMN_VISIBILITY: Record<string, ColumnVisibilityConfig> = {
  workerId: { fixedWidth: 12, priority: 0 },
  status: { fixedWidth: 6, priority: 1 },
  currentRolloutId: { fixedWidth: 14, priority: 3 },
  currentAttemptId: { fixedWidth: 14, priority: 3 },
  lastHeartbeatTime: { fixedWidth: 10, priority: 2 },
  lastBusyTime: { fixedWidth: 10, priority: 3 },
  lastIdleTime: { fixedWidth: 10, priority: 3 },
  lastDequeueTime: { fixedWidth: 10, priority: 1 },
  actions: { fixedWidth: 5, priority: 0 },
};

export type WorkersTableRecord = Worker & {
  timestamps: Record<
    'lastHeartbeatTime' | 'lastBusyTime' | 'lastIdleTime' | 'lastDequeueTime',
    { absolute: string; relative: string }
  >;
};

const buildTimestampMeta = (value: Worker['lastHeartbeatTime']) => ({
  absolute: formatDateTime(value),
  relative: formatRelativeTime(value),
});

function buildWorkerRecord(worker: Worker): WorkersTableRecord {
  return {
    ...worker,
    timestamps: {
      lastHeartbeatTime: buildTimestampMeta(worker.lastHeartbeatTime),
      lastBusyTime: buildTimestampMeta(worker.lastBusyTime),
      lastIdleTime: buildTimestampMeta(worker.lastIdleTime),
      lastDequeueTime: buildTimestampMeta(worker.lastDequeueTime),
    },
  };
}

type WorkersColumnsOptions = {
  onShowDetails: (worker: Worker) => void;
};

const STATUS_COLORS: Record<Worker['status'], string> = {
  busy: 'orange',
  idle: 'teal',
  unknown: 'gray',
};

function createWorkersColumns({ onShowDetails }: WorkersColumnsOptions): DataTableColumn<WorkersTableRecord>[] {
  return [
    {
      accessor: 'workerId',
      title: 'Runner ID',
      sortable: true,
      render: ({ workerId }) => (
        <Group gap={2} wrap='nowrap'>
          <Text fw={500} size='sm'>
            {workerId}
          </Text>
          <CopyButton value={workerId}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
                <ActionIcon
                  aria-label={`Copy worker ID ${workerId}`}
                  variant='subtle'
                  color={copied ? 'teal' : 'gray'}
                  size='sm'
                  onClick={(event) => {
                    event.stopPropagation();
                    copy();
                  }}
                >
                  {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                </ActionIcon>
              </Tooltip>
            )}
          </CopyButton>
        </Group>
      ),
    },
    {
      accessor: 'status',
      title: 'Status',
      sortable: true,
      render: ({ status }) => {
        const color = STATUS_COLORS[status] ?? 'gray';
        return (
          <Badge size='sm' variant='light' color={color} radius='sm'>
            {formatStatusLabel(status)}
          </Badge>
        );
      },
    },
    {
      accessor: 'currentRolloutId',
      title: 'Current Rollout',
      sortable: true,
      render: ({ currentRolloutId }) => <Text size='sm'>{currentRolloutId ?? '—'}</Text>,
    },
    {
      accessor: 'currentAttemptId',
      title: 'Current Attempt',
      sortable: true,
      render: ({ currentAttemptId }) => <Text size='sm'>{currentAttemptId ?? '—'}</Text>,
    },
    {
      accessor: 'lastHeartbeatTime',
      title: 'Heartbeat',
      sortable: true,
      render: ({ timestamps }) => (
        <Stack gap={0} justify='center'>
          <Text size='sm'>{timestamps.lastHeartbeatTime.relative}</Text>
          {timestamps.lastHeartbeatTime.absolute !== '—' && (
            <Text size='xs' c='dimmed'>
              {timestamps.lastHeartbeatTime.absolute}
            </Text>
          )}
        </Stack>
      ),
    },
    {
      accessor: 'lastBusyTime',
      title: 'Last Busy',
      sortable: true,
      render: ({ timestamps }) => (
        <Stack gap={0} justify='center'>
          <Text size='sm'>{timestamps.lastBusyTime.relative}</Text>
          {timestamps.lastBusyTime.absolute !== '—' && (
            <Text size='xs' c='dimmed'>
              {timestamps.lastBusyTime.absolute}
            </Text>
          )}
        </Stack>
      ),
    },
    {
      accessor: 'lastIdleTime',
      title: 'Last Idle',
      sortable: true,
      render: ({ timestamps }) => (
        <Stack gap={0} justify='center'>
          <Text size='sm'>{timestamps.lastIdleTime.relative}</Text>
          {timestamps.lastIdleTime.absolute !== '—' && (
            <Text size='xs' c='dimmed'>
              {timestamps.lastIdleTime.absolute}
            </Text>
          )}
        </Stack>
      ),
    },
    {
      accessor: 'lastDequeueTime',
      title: 'Last Dequeue',
      sortable: true,
      render: ({ timestamps }) => (
        <Stack gap={0} justify='center'>
          <Text size='sm'>{timestamps.lastDequeueTime.relative}</Text>
          {timestamps.lastDequeueTime.absolute !== '—' && (
            <Text size='xs' c='dimmed'>
              {timestamps.lastDequeueTime.absolute}
            </Text>
          )}
        </Stack>
      ),
    },
    {
      accessor: 'actions',
      title: 'Actions',
      textAlign: 'left',
      render: (record) => (
        <Tooltip label='Show runner detail' withArrow disabled={!onShowDetails}>
          <ActionIcon
            aria-label='Show runner detail'
            variant='subtle'
            color='gray'
            onClick={(event) => {
              event.stopPropagation();
              onShowDetails(record);
            }}
          >
            <IconInfoCircle size={16} />
          </ActionIcon>
        </Tooltip>
      ),
    },
  ];
}

export type WorkersTableProps = {
  workers: Worker[] | undefined;
  totalRecords: number;
  isFetching: boolean;
  isError: boolean;
  error: unknown;
  searchTerm: string;
  sort: { column: string; direction: 'asc' | 'desc' };
  page: number;
  recordsPerPage: number;
  onSortStatusChange: (status: DataTableSortStatus<WorkersTableRecord>) => void;
  onPageChange: (page: number) => void;
  onRecordsPerPageChange: (value: number) => void;
  onResetFilters: () => void;
  onRefetch: () => void;
  onShowDetails: (worker: Worker) => void;
  recordsPerPageOptions?: number[];
};

export function WorkersTable({
  workers,
  totalRecords,
  isFetching,
  isError,
  error,
  searchTerm,
  sort,
  page,
  recordsPerPage,
  onSortStatusChange,
  onPageChange,
  onRecordsPerPageChange,
  onResetFilters,
  onRefetch,
  onShowDetails,
  recordsPerPageOptions = DEFAULT_RECORDS_PER_PAGE_OPTIONS,
}: WorkersTableProps) {
  const { ref: tableContainerRef, width: containerWidth } = useElementSize();
  const { width: viewportWidth } = useViewportSize();

  const layoutAwareContainerWidth = useMemo(
    () => getLayoutAwareWidth(containerWidth, viewportWidth),
    [containerWidth, viewportWidth],
  );

  const workerRecords = useMemo<WorkersTableRecord[]>(() => {
    if (!workers) {
      return [];
    }
    return workers.map((worker) => buildWorkerRecord(worker));
  }, [workers]);

  const columns = useMemo(() => createWorkersColumns({ onShowDetails }), [onShowDetails]);
  const responsiveColumns = useMemo(
    () => createResponsiveColumns(columns, layoutAwareContainerWidth, COLUMN_VISIBILITY),
    [columns, layoutAwareContainerWidth],
  );

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(Math.max(0, totalRecords) / Math.max(1, recordsPerPage))),
    [recordsPerPage, totalRecords],
  );

  useEffect(() => {
    if (page > totalPages) {
      onPageChange(totalPages);
    }
  }, [onPageChange, page, totalPages]);

  const hasActiveFilters = searchTerm.trim().length > 0;

  const sortStatus: DataTableSortStatus<WorkersTableRecord> = {
    columnAccessor: sort.column,
    direction: sort.direction,
  };

  const handleSortStatusChange = useCallback(
    (status: DataTableSortStatus<WorkersTableRecord>) => {
      onSortStatusChange(status);
    },
    [onSortStatusChange],
  );

  const errorDescriptor = isError ? getErrorDescriptor(error) : null;
  const errorMessage = isError
    ? `Workers are temporarily unavailable${errorDescriptor ? ` (${errorDescriptor})` : ''}.`
    : 'Workers are temporarily unavailable.';

  const emptyState = (
    <Stack gap='sm' align='center' py='lg'>
      {isError ? (
        <>
          <Text fw={600} size='sm'>
            {errorMessage}
          </Text>
          <Text size='sm' c='dimmed' ta='center'>
            Use the retry button to try again, or adjust the search to broaden the results.
          </Text>
          <Group gap='xs'>
            <Button size='xs' variant='light' color='gray' leftSection={<IconRefresh size={14} />} onClick={onRefetch}>
              Retry
            </Button>
            {hasActiveFilters ? (
              <Button size='xs' variant='subtle' onClick={onResetFilters}>
                Clear filters
              </Button>
            ) : null}
          </Group>
        </>
      ) : (
        <>
          <Text fw={600} size='sm'>
            No workers found
          </Text>
          <Text size='sm' c='dimmed' ta='center'>
            {hasActiveFilters
              ? 'Try adjusting the search to see more results.'
              : 'Try refreshing to fetch the latest worker status.'}
          </Text>
          <Group gap='xs'>
            <Button size='xs' variant='light' leftSection={<IconRefresh size={14} />} onClick={onRefetch}>
              Refresh
            </Button>
            {hasActiveFilters ? (
              <Button size='xs' variant='subtle' onClick={onResetFilters}>
                Clear filters
              </Button>
            ) : null}
          </Group>
        </>
      )}
    </Stack>
  );

  return (
    <Box ref={tableContainerRef}>
      <DataTable<WorkersTableRecord>
        classNames={{ root: 'workers-table' }}
        withTableBorder
        withColumnBorders
        highlightOnHover
        verticalAlign='center'
        minHeight={workerRecords.length === 0 ? 400 : undefined}
        idAccessor='workerId'
        records={workerRecords}
        columns={responsiveColumns}
        totalRecords={totalRecords}
        recordsPerPage={recordsPerPage}
        page={page}
        onPageChange={onPageChange}
        onRecordsPerPageChange={onRecordsPerPageChange}
        recordsPerPageOptions={recordsPerPageOptions}
        sortStatus={sortStatus}
        onSortStatusChange={handleSortStatusChange}
        fetching={isFetching}
        loaderSize='sm'
        emptyState={workerRecords.length === 0 ? emptyState : undefined}
      />
    </Box>
  );
}
