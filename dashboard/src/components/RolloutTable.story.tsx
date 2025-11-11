// Copyright (c) Microsoft. All rights reserved.

import { useMemo, useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { IconSearch } from '@tabler/icons-react';
import { Box, Stack, TextInput, Title } from '@mantine/core';
import type { RolloutsSortState } from '@/features/rollouts';
import type { Rollout, RolloutMode, RolloutStatus } from '@/types';
import { compareRecords } from '@/utils/table';
import { STORY_DATE_NOW_SECONDS } from '../../.storybook/constants';
import { buildRolloutRecord, RolloutTable, type RolloutTableRecord } from './RolloutTable.component';

const meta: Meta<typeof RolloutTable> = {
  title: 'Components/RolloutTable',
  component: RolloutTable,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof RolloutTable>;

const now = STORY_DATE_NOW_SECONDS;

const sampleRollouts: Rollout[] = [
  {
    rolloutId: 'ro-story-001',
    input: { task: 'Generate onboarding summary' },
    startTime: now - 3200,
    endTime: null,
    mode: 'train',
    resourcesId: 'rs-story-001',
    status: 'running',
    config: { retries: 1 },
    metadata: { owner: 'alice' },
    attempt: {
      rolloutId: 'ro-story-001',
      attemptId: 'at-story-010',
      sequenceId: 1,
      startTime: now - 3200,
      endTime: null,
      status: 'running',
      workerId: 'worker-east',
      lastHeartbeatTime: now - 45,
      metadata: { info: 'Worker is processing' },
    },
  },
  {
    rolloutId: 'ro-story-002',
    input: { task: 'Classify feedback tickets' },
    startTime: now - 7200,
    endTime: now - 5400,
    mode: 'val',
    resourcesId: 'rs-story-002',
    status: 'succeeded',
    config: { retries: 2 },
    metadata: { owner: 'bob' },
    attempt: {
      rolloutId: 'ro-story-002',
      attemptId: 'at-story-011',
      sequenceId: 2,
      startTime: now - 6200,
      endTime: now - 5400,
      status: 'succeeded',
      workerId: 'worker-north',
      lastHeartbeatTime: now - 5400,
      metadata: { previousAttempt: 'at-story-010' },
    },
  },
  {
    rolloutId: 'ro-story-003',
    input: { task: 'Analyze experiment results' },
    startTime: now - 10800,
    endTime: now - 9600,
    mode: 'test',
    resourcesId: 'rs-story-003',
    status: 'failed',
    config: { retries: 1 },
    metadata: { owner: 'carol' },
    attempt: {
      rolloutId: 'ro-story-003',
      attemptId: 'at-story-012',
      sequenceId: 3,
      startTime: now - 10200,
      endTime: now - 9600,
      status: 'failed',
      workerId: 'worker-west',
      lastHeartbeatTime: now - 9600,
      metadata: { reason: 'Timeout' },
    },
  },
  {
    rolloutId: 'ro-story-004',
    input: { task: 'Evaluate prompt variants' },
    startTime: now - 3600,
    endTime: null,
    mode: 'train',
    resourcesId: null,
    status: 'preparing',
    config: { retries: 0 },
    metadata: { owner: 'dave' },
    attempt: null,
  },
  {
    rolloutId: 'ro-story-005',
    input: { task: 'Generate quick answers' },
    startTime: now - 1800,
    endTime: null,
    mode: 'val',
    resourcesId: 'rs-story-004',
    status: 'running',
    config: { retries: 0 },
    metadata: { owner: 'eva' },
    attempt: {
      rolloutId: 'ro-story-005',
      attemptId: 'at-story-013',
      sequenceId: 1,
      startTime: now - 1800,
      endTime: null,
      status: 'running',
      workerId: null,
      lastHeartbeatTime: now - 75,
      metadata: null,
    },
  },
  {
    rolloutId: 'ro-story-006',
    input: { task: 'Compile release notes' },
    startTime: now - 9600,
    endTime: now - 9000,
    mode: null,
    resourcesId: 'rs-story-005',
    status: 'cancelled',
    config: { retries: 3 },
    metadata: null,
    attempt: {
      rolloutId: 'ro-story-006',
      attemptId: 'at-story-014',
      sequenceId: 1,
      startTime: now - 9600,
      endTime: now - 9000,
      status: 'timeout',
      workerId: 'worker-south',
      lastHeartbeatTime: now - 9000,
      metadata: { info: 'Cancelled by operator' },
    },
  },
];

type WrapperProps = {
  maxWidth: number;
  rollouts?: Rollout[] | undefined;
  isFetching?: boolean;
  isError?: boolean;
  error?: unknown;
};

function RolloutTableStoryWrapper({
  maxWidth,
  rollouts = sampleRollouts,
  isFetching = false,
  isError = false,
  error = null,
}: WrapperProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilters, setStatusFilters] = useState<RolloutStatus[]>([]);
  const [modeFilters, setModeFilters] = useState<RolloutMode[]>([]);
  const [page, setPage] = useState(1);
  const [recordsPerPage, setRecordsPerPage] = useState(5);
  const [sort, setSort] = useState<RolloutsSortState>({
    column: 'startTimestamp',
    direction: 'desc',
  });

  const tableRecords = useMemo<RolloutTableRecord[]>(() => {
    if (!rollouts) {
      return [];
    }
    return rollouts.map((rolloutItem) => buildRolloutRecord(rolloutItem));
  }, [rollouts]);

  const filteredRecords = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();
    return tableRecords.filter((record) => {
      const matchesSearch = normalizedSearch.length === 0 || record.rolloutId.toLowerCase().includes(normalizedSearch);
      const matchesStatus = statusFilters.length === 0 || statusFilters.includes(record.status);
      const matchesMode = modeFilters.length === 0 || (record.mode !== null && modeFilters.includes(record.mode));
      return matchesSearch && matchesStatus && matchesMode;
    });
  }, [modeFilters, searchTerm, statusFilters, tableRecords]);

  const sortedRecords = useMemo(() => {
    const sorted = filteredRecords.slice();
    if (!sorted.length) {
      return sorted;
    }
    const comparatorKey = sort.column as keyof RolloutTableRecord;
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

  const pagedRollouts = useMemo(() => pagedRecords.map((record) => record as Rollout), [pagedRecords]);

  return (
    <Box mx='auto' style={{ maxWidth, width: '100%', padding: 16 }}>
      <Stack gap='md'>
        <Title order={2}>Rollouts</Title>
        <TextInput
          placeholder='Search by Rollout ID'
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.currentTarget.value)}
          leftSection={<IconSearch size={16} />}
          data-testid='rollouts-search-input'
          w='100%'
          style={{ maxWidth: 360 }}
        />
        <RolloutTable
          rollouts={pagedRollouts}
          totalRecords={totalRecordsValue}
          isFetching={isFetching}
          isError={isError}
          error={error}
          searchTerm={searchTerm}
          statusFilters={statusFilters}
          modeFilters={modeFilters}
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onStatusFilterChange={(values) => {
            setStatusFilters(values);
            setPage(1);
          }}
          onStatusFilterReset={() => {
            setStatusFilters([]);
            setPage(1);
          }}
          onModeFilterChange={(values) => {
            setModeFilters(values);
            setPage(1);
          }}
          onModeFilterReset={() => {
            setModeFilters([]);
            setPage(1);
          }}
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
            setStatusFilters([]);
            setModeFilters([]);
            setSort({ column: 'startTimestamp', direction: 'desc' });
            setPage(1);
          }}
          onRefetch={() => undefined}
          recordsPerPageOptions={[5, 10, 20]}
        />
      </Stack>
    </Box>
  );
}

export const WideContainer: Story = {
  render: () => <RolloutTableStoryWrapper maxWidth={1280} />,
};

export const MediumContainer: Story = {
  render: () => <RolloutTableStoryWrapper maxWidth={960} />,
};

export const NarrowContainer: Story = {
  render: () => <RolloutTableStoryWrapper maxWidth={720} />,
};

export const DrawerWidth: Story = {
  render: () => <RolloutTableStoryWrapper maxWidth={520} />,
};

export const ErrorState: Story = {
  render: () => (
    <RolloutTableStoryWrapper maxWidth={600} rollouts={[]} isError error={new Error('Network unreachable')} />
  ),
};
