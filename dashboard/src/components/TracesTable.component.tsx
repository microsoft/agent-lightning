// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect, useMemo } from 'react';
import {
  IconAlertCircle,
  IconCheck,
  IconCopy,
  IconFileDescription,
  IconRefresh,
  IconRouteSquare,
} from '@tabler/icons-react';
import { DataTable, type DataTableColumn, type DataTableSortStatus } from 'mantine-datatable';
import { ActionIcon, Badge, Box, Button, CopyButton, Group, Stack, Text, Tooltip } from '@mantine/core';
import { useElementSize, useViewportSize } from '@mantine/hooks';
import { getLayoutAwareWidth } from '@/layouts/helper';
import type { Span } from '@/types';
import { getErrorDescriptor } from '@/utils/error';
import { formatDateTimeWithMilliseconds, formatDuration, toTimestamp } from '@/utils/format';
import { createResponsiveColumns, type ColumnVisibilityConfig } from '@/utils/table';

const DEFAULT_RECORDS_PER_PAGE_OPTIONS = [50, 100, 200, 500];

const COLUMN_VISIBILITY: Record<string, ColumnVisibilityConfig> = {
  name: { minWidth: 12.5, priority: 0 },
  sequenceId: { fixedWidth: 6, priority: 1 },
  spanId: { fixedWidth: 14, priority: 1 },
  traceId: { fixedWidth: 24, priority: 3 },
  parentId: { fixedWidth: 12, priority: 2 },
  statusCode: { fixedWidth: 8, priority: 2 },
  attributeKeys: { minWidth: 12.5, priority: 2 },
  startTime: { fixedWidth: 15, priority: 1 },
  endTime: { fixedWidth: 15, priority: 1 },
  duration: { fixedWidth: 10, priority: 3 },
  actionsPlaceholder: { fixedWidth: 6, priority: 0 },
};

const STATUS_COLORS: Record<string, string> = {
  UNSET: 'gray',
  OK: 'teal',
  ERROR: 'red',
};

export type TracesTableRecord = Span & {
  statusCode: string;
  attributeKeys: string;
  duration: number;
  actionsPlaceholder?: null;
};

export function buildTraceRecord(span: Span): TracesTableRecord {
  const statusCode = span.status.status_code;
  const attributeKeys = Object.keys(span.attributes ?? {}).join(', ') || '';
  const startTimestamp = toTimestamp(span.startTime);
  const endTimestamp = toTimestamp(span.endTime);
  const duration = endTimestamp && startTimestamp ? endTimestamp - startTimestamp : 0;

  return {
    ...span,
    statusCode,
    attributeKeys,
    duration,
    actionsPlaceholder: null,
  };
}

type TracesColumnsOptions = {
  onShowRollout?: (record: TracesTableRecord) => void;
  onShowSpanDetail?: (record: TracesTableRecord) => void;
  onParentIdClick?: (parentId: string) => void;
  spanIds: Set<string>;
};

function createTracesColumns({
  onShowRollout,
  onShowSpanDetail,
  onParentIdClick,
  spanIds,
}: TracesColumnsOptions): DataTableColumn<TracesTableRecord>[] {
  return [
    {
      accessor: 'name',
      title: 'Name',
      sortable: true,
      render: ({ name }) => (
        <Text size='sm' fw={500}>
          {name}
        </Text>
      ),
    },
    {
      accessor: 'sequenceId',
      title: 'Seq.',
      sortable: true,
      render: ({ sequenceId }) => <Text size='sm'>{sequenceId}</Text>,
    },
    {
      accessor: 'traceId',
      title: 'Trace ID',
      sortable: true,
      render: ({ traceId }) => (
        <Group gap={2}>
          <Text size='sm'>{traceId}</Text>
          <CopyButton value={traceId}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
                <ActionIcon
                  aria-label={`Copy trace ID ${traceId}`}
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
      accessor: 'spanId',
      title: 'Span ID',
      sortable: true,
      render: ({ spanId }) => (
        <Group gap={2}>
          <Text size='sm'>{spanId}</Text>
          <CopyButton value={spanId}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
                <ActionIcon
                  aria-label={`Copy span ID ${spanId}`}
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
      accessor: 'parentId',
      title: 'Parent ID',
      sortable: true,
      render: ({ parentId }) => {
        if (!parentId) {
          return (
            <Text size='sm' c='dimmed'>
              —
            </Text>
          );
        }

        const parentExists = spanIds.has(parentId);
        const isInteractive = parentExists && typeof onParentIdClick === 'function';

        return (
          <Group gap={2}>
            <Text
              size='sm'
              c={parentExists ? undefined : 'red'}
              style={{ cursor: isInteractive ? 'pointer' : undefined }}
              onClick={(event) => {
                if (isInteractive) {
                  event.stopPropagation();
                  onParentIdClick?.(parentId);
                }
              }}
            >
              {parentId.slice(0, 8)}
            </Text>
            {!parentExists && (
              <Tooltip label='Parent span not found in table' withArrow>
                <IconAlertCircle size={14} color='red' />
              </Tooltip>
            )}
          </Group>
        );
      },
    },

    {
      accessor: 'statusCode',
      title: 'Status',
      sortable: true,
      render: ({ statusCode }) => (
        <Badge size='sm' variant='light' color={STATUS_COLORS[statusCode] ?? 'gray'}>
          {statusCode}
        </Badge>
      ),
    },
    {
      accessor: 'attributeKeys',
      title: 'Attribute Keys',
      render: ({ attributeKeys }) =>
        attributeKeys ? (
          <Text size='sm' lineClamp={1}>
            {/* TODO: dim "." and "," and other characters are just normal text */}
            {attributeKeys}
          </Text>
        ) : (
          <Text size='sm' c='dimmed'>
            —
          </Text>
        ),
    },
    {
      accessor: 'startTime',
      title: 'Start Time',
      sortable: true,
      textAlign: 'left',
      render: ({ startTime }) => <Text size='sm'>{formatDateTimeWithMilliseconds(toTimestamp(startTime))}</Text>,
    },
    {
      accessor: 'endTime',
      title: 'End Time',
      sortable: true,
      textAlign: 'left',
      render: ({ endTime }) => <Text size='sm'>{formatDateTimeWithMilliseconds(toTimestamp(endTime))}</Text>,
    },
    {
      accessor: 'duration',
      title: 'Duration',
      sortable: true,
      textAlign: 'left',
      render: ({ duration }) => <Text size='sm'>{formatDuration(duration)}</Text>,
    },
    {
      accessor: 'actionsPlaceholder',
      title: 'Actions',
      render: (record) => (
        <Group gap={2}>
          <Tooltip label='Show rollout' withArrow disabled={!onShowRollout}>
            <ActionIcon
              aria-label='Show rollout'
              variant='subtle'
              color='gray'
              onClick={(event) => {
                event.stopPropagation();
                onShowRollout?.(record);
              }}
            >
              <IconRouteSquare size={16} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label='Show span detail' withArrow disabled={!onShowSpanDetail}>
            <ActionIcon
              aria-label='Show span detail'
              variant='subtle'
              color='gray'
              onClick={(event) => {
                event.stopPropagation();
                onShowSpanDetail?.(record);
              }}
            >
              <IconFileDescription size={16} />
            </ActionIcon>
          </Tooltip>
        </Group>
      ),
    },
  ];
}

export type TracesTableProps = {
  spans: Span[] | undefined;
  totalRecords: number;
  isFetching: boolean;
  isError: boolean;
  error: unknown;
  selectionMessage?: string;
  searchTerm: string;
  sort: { column: string; direction: 'asc' | 'desc' };
  page: number;
  recordsPerPage: number;
  onSortStatusChange: (status: DataTableSortStatus<TracesTableRecord>) => void;
  onPageChange: (page: number) => void;
  onRecordsPerPageChange: (value: number) => void;
  onResetFilters: () => void;
  onRefetch: () => void;
  onShowRollout?: (record: TracesTableRecord) => void;
  onShowSpanDetail?: (record: TracesTableRecord) => void;
  onParentIdClick?: (parentId: string) => void;
  recordsPerPageOptions?: number[];
};

export function TracesTable({
  spans,
  totalRecords,
  isFetching,
  isError,
  error,
  selectionMessage,
  searchTerm,
  sort,
  page,
  recordsPerPage,
  onSortStatusChange,
  onPageChange,
  onRecordsPerPageChange,
  onResetFilters,
  onRefetch,
  onShowRollout,
  onShowSpanDetail,
  onParentIdClick,
  recordsPerPageOptions = DEFAULT_RECORDS_PER_PAGE_OPTIONS,
}: TracesTableProps) {
  const { ref: tableContainerRef, width: containerWidth } = useElementSize();
  const { width: viewportWidth } = useViewportSize();

  const traceRecords = useMemo<TracesTableRecord[]>(() => {
    if (!spans) {
      return [];
    }
    return spans.map((span) => buildTraceRecord(span));
  }, [spans]);

  const spanIds = useMemo(() => {
    return new Set(traceRecords.map((record) => record.spanId));
  }, [traceRecords]);

  const columns = useMemo(
    () =>
      createTracesColumns({
        onShowRollout,
        onShowSpanDetail,
        onParentIdClick,
        spanIds,
      }),
    [onShowRollout, onShowSpanDetail, onParentIdClick, spanIds],
  );

  const layoutAwareContainerWidth = useMemo(
    () => getLayoutAwareWidth(containerWidth, viewportWidth),
    [containerWidth, viewportWidth],
  );

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

  const sortStatus: DataTableSortStatus<TracesTableRecord> = {
    columnAccessor: sort.column,
    direction: sort.direction,
  };

  const handleSortStatusChange = useCallback(
    (status: DataTableSortStatus<TracesTableRecord>) => {
      onSortStatusChange(status);
    },
    [onSortStatusChange],
  );

  const errorDescriptor = isError ? getErrorDescriptor(error) : null;
  const errorMessage = isError
    ? `Traces are temporarily unavailable${errorDescriptor ? ` (${errorDescriptor})` : ''}.`
    : 'Traces are temporarily unavailable.';

  const selectionEmptyState = selectionMessage ? (
    <Stack gap='sm' align='center' py='xl'>
      <Text fw={600} size='sm'>
        {selectionMessage}
      </Text>
      <Text size='sm' c='dimmed' ta='center'>
        Choose a rollout and attempt from the controls above to load trace results.
      </Text>
    </Stack>
  ) : null;

  const fallbackEmptyState = (
    <Stack gap='sm' align='center' py='lg'>
      {isError ? (
        <>
          <Text fw={600} size='sm'>
            {errorMessage}
          </Text>
          <Text size='sm' c='dimmed' ta='center'>
            Use the retry button to try again, or adjust the filters to broaden the results.
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
            No traces found
          </Text>
          <Text size='sm' c='dimmed' ta='center'>
            {hasActiveFilters
              ? 'Try adjusting the search to see more results.'
              : 'Try refreshing to fetch the latest traces.'}
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

  const emptyState = selectionEmptyState ?? fallbackEmptyState;

  return (
    <Box ref={tableContainerRef}>
      <DataTable<TracesTableRecord>
        classNames={{ root: 'traces-table' }}
        withTableBorder
        withColumnBorders
        highlightOnHover
        verticalAlign='center'
        minHeight={traceRecords.length === 0 ? 500 : undefined}
        idAccessor='spanId'
        records={traceRecords}
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
        emptyState={traceRecords.length === 0 ? emptyState : undefined}
      />
    </Box>
  );
}
