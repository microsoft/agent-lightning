// Copyright (c) Microsoft. All rights reserved.

import { useMemo, useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { IconSearch } from '@tabler/icons-react';
import { Box, Stack, TextInput, Title } from '@mantine/core';
import type { Span } from '@/types';
import { compareRecords } from '@/utils/table';
import { buildTraceRecord, TracesTable, type TracesTableRecord } from './TracesTable.component';

const meta: Meta<typeof TracesTable> = {
  title: 'Components/TracesTable',
  component: TracesTable,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof TracesTable>;

const now = Math.floor(1762775145209 / 1000);

const sampleSpans: Span[] = [
  {
    rolloutId: 'ro-trace-001',
    attemptId: 'at-trace-001',
    sequenceId: 1,
    traceId: 'trace-abc123def456',
    spanId: 'span-root-001',
    parentId: null,
    name: 'main_task',
    status: { status_code: 'OK', description: null },
    attributes: {
      'task.type': 'generation',
      'task.priority': 'high',
      'user.id': 'user-123',
    },
    startTime: now - 100,
    endTime: now - 10,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-001',
    attemptId: 'at-trace-001',
    sequenceId: 1,
    traceId: 'trace-abc123def456',
    spanId: 'span-child-001',
    parentId: 'span-root-001',
    name: 'llm_call',
    status: { status_code: 'OK', description: null },
    attributes: {
      'llm.model': 'gpt-4',
      'llm.temperature': 0.7,
      'llm.max_tokens': 2048,
    },
    startTime: now - 90,
    endTime: now - 50,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-001',
    attemptId: 'at-trace-001',
    sequenceId: 1,
    traceId: 'trace-abc123def456',
    spanId: 'span-child-002',
    parentId: 'span-root-001',
    name: 'database_query',
    status: { status_code: 'OK', description: null },
    attributes: {
      'db.system': 'postgresql',
      'db.operation': 'SELECT',
      'db.table': 'users',
    },
    startTime: now - 80,
    endTime: now - 70,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-002',
    attemptId: 'at-trace-002',
    sequenceId: 1,
    traceId: 'trace-xyz789ghi012',
    spanId: 'span-error-001',
    parentId: 'span-missing-parent',
    name: 'failed_operation',
    status: { status_code: 'ERROR', description: 'Connection timeout' },
    attributes: {
      'error.type': 'TimeoutError',
      'error.message': 'Connection timed out after 30s',
    },
    startTime: now - 150,
    endTime: now - 120,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-003',
    attemptId: 'at-trace-003',
    sequenceId: 1,
    traceId: 'trace-unset123',
    spanId: 'span-unset-001',
    parentId: null,
    name: 'pending_task',
    status: { status_code: 'UNSET', description: null },
    attributes: {},
    startTime: now - 30,
    endTime: now - 5,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-004',
    attemptId: 'at-trace-004',
    sequenceId: 2,
    traceId: 'trace-nested456',
    spanId: 'span-parent-001',
    parentId: null,
    name: 'workflow_execution',
    status: { status_code: 'OK', description: null },
    attributes: {
      'workflow.name': 'data_processing',
      'workflow.version': '2.1.0',
    },
    startTime: now - 200,
    endTime: now - 50,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-004',
    attemptId: 'at-trace-004',
    sequenceId: 2,
    traceId: 'trace-nested456',
    spanId: 'span-child-nested-001',
    parentId: 'span-parent-001',
    name: 'step_1_validation',
    status: { status_code: 'OK', description: null },
    attributes: {
      'step.name': 'validation',
      'step.index': 1,
    },
    startTime: now - 195,
    endTime: now - 180,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
  {
    rolloutId: 'ro-trace-004',
    attemptId: 'at-trace-004',
    sequenceId: 2,
    traceId: 'trace-nested456',
    spanId: 'span-child-nested-002',
    parentId: 'span-parent-001',
    name: 'step_2_processing',
    status: { status_code: 'ERROR', description: 'Validation failed' },
    attributes: {
      'step.name': 'processing',
      'step.index': 2,
      'error.type': 'ValidationError',
    },
    startTime: now - 175,
    endTime: now - 160,
    events: [],
    links: [],
    context: {},
    parent: null,
    resource: {},
  },
];

type WrapperProps = {
  maxWidth: number;
  spans?: Span[] | undefined;
  isFetching?: boolean;
  isError?: boolean;
  error?: unknown;
};

function TracesTableStoryWrapper({
  maxWidth,
  spans = sampleSpans,
  isFetching = false,
  isError = false,
  error = null,
}: WrapperProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [recordsPerPage, setRecordsPerPage] = useState(10);
  const [sort, setSort] = useState<{ column: string; direction: 'asc' | 'desc' }>({
    column: 'startTime',
    direction: 'desc',
  });

  const tableRecords = useMemo<TracesTableRecord[]>(() => {
    if (!spans) {
      return [];
    }
    return spans.map((span) => buildTraceRecord(span));
  }, [spans]);

  const filteredRecords = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();
    if (normalizedSearch.length === 0) {
      return tableRecords;
    }
    return tableRecords.filter(
      (record) =>
        record.traceId.toLowerCase().includes(normalizedSearch) ||
        record.spanId.toLowerCase().includes(normalizedSearch) ||
        record.name.toLowerCase().includes(normalizedSearch),
    );
  }, [searchTerm, tableRecords]);

  const sortedRecords = useMemo(() => {
    const sorted = filteredRecords.slice();
    if (!sorted.length) {
      return sorted;
    }
    const comparatorKey = sort.column as keyof TracesTableRecord;
    if (!(comparatorKey in sorted[0])) {
      return sorted;
    }
    sorted.sort((a, b) => compareRecords(a, b, comparatorKey));
    if (sort.direction === 'desc') {
      sorted.reverse();
    }
    return sorted;
  }, [filteredRecords, sort]);

  const totalRecordsValue = sortedRecords.length;

  const pagedRecords = useMemo(() => {
    const startIndex = (page - 1) * recordsPerPage;
    const endIndex = startIndex + recordsPerPage;
    return sortedRecords.slice(startIndex, endIndex);
  }, [page, recordsPerPage, sortedRecords]);

  const pagedSpans = useMemo(() => pagedRecords.map((record) => record as Span), [pagedRecords]);

  const handleShowRollout = (record: any) => {
    console.log('Show rollout for:', record.rolloutId);
  };

  const handleShowSpanDetail = (record: any) => {
    console.log('Show span detail for:', record.spanId, record);
  };

  const handleParentIdClick = (parentId: string) => {
    console.log('Navigate to parent span:', parentId);
    setSearchTerm(parentId);
  };

  return (
    <Box mx='auto' style={{ maxWidth, width: '100%', padding: 16 }}>
      <Stack gap='md'>
        <Title order={2}>Traces</Title>
        <TextInput
          placeholder='Search by Trace ID, Span ID, or Name'
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.currentTarget.value)}
          leftSection={<IconSearch size={16} />}
          data-testid='traces-search-input'
          w='100%'
          style={{ maxWidth: 360 }}
        />
        <TracesTable
          spans={pagedSpans}
          totalRecords={totalRecordsValue}
          isFetching={isFetching}
          isError={isError}
          error={error}
          searchTerm={searchTerm}
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onSortStatusChange={(status) => {
            setSort({
              column: status.columnAccessor as string,
              direction: status.direction,
            });
          }}
          onPageChange={setPage}
          onRecordsPerPageChange={(value) => {
            setRecordsPerPage(value);
            setPage(1);
          }}
          onResetFilters={() => {
            setSearchTerm('');
            setSort({ column: 'startTime', direction: 'desc' });
            setPage(1);
          }}
          onRefetch={() => undefined}
          onShowRollout={handleShowRollout}
          onShowSpanDetail={handleShowSpanDetail}
          onParentIdClick={handleParentIdClick}
          recordsPerPageOptions={[10, 20, 50]}
        />
      </Stack>
    </Box>
  );
}

export const WideContainer: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={1400} />,
};

export const MediumContainer: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={960} />,
};

export const NarrowContainer: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={720} />,
};

export const DrawerWidth: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={520} />,
};

export const ErrorState: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={960} spans={[]} isError error={new Error('Network unreachable')} />,
};

export const LoadingState: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={960} spans={[]} isFetching />,
};

export const EmptyState: Story = {
  render: () => <TracesTableStoryWrapper maxWidth={960} spans={[]} />,
};

export const WithMissingParent: Story = {
  render: () => (
    <TracesTableStoryWrapper maxWidth={1200} spans={sampleSpans.filter((s) => s.spanId === 'span-error-001')} />
  ),
};

export const NestedSpans: Story = {
  render: () => (
    <TracesTableStoryWrapper maxWidth={1200} spans={sampleSpans.filter((s) => s.traceId === 'trace-nested456')} />
  ),
};
