// Copyright (c) Microsoft. All rights reserved.

import { useMemo, useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { IconSearch } from '@tabler/icons-react';
import { Box, Stack, TextInput, Title } from '@mantine/core';
import type { Worker } from '@/types';
import { WorkersTable } from './WorkersTable.component';

const meta: Meta<typeof WorkersTable> = {
  title: 'Components/WorkersTable',
  component: WorkersTable,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof WorkersTable>;

const now = Math.floor(Date.now() / 1000);

const sampleWorkers: Worker[] = [
  {
    workerId: 'worker-east',
    status: 'busy',
    heartbeatStats: { queueDepth: 2, gpuUtilization: 0.82 },
    lastHeartbeatTime: now - 20,
    lastDequeueTime: now - 60,
    lastBusyTime: now - 120,
    lastIdleTime: now - 600,
    currentRolloutId: 'ro-story-001',
    currentAttemptId: 'at-story-010',
  },
  {
    workerId: 'worker-west',
    status: 'busy',
    heartbeatStats: { queueDepth: 1 },
    lastHeartbeatTime: now - 45,
    lastDequeueTime: now - 300,
    lastBusyTime: now - 200,
    lastIdleTime: now - 4800,
    currentRolloutId: 'ro-story-003',
    currentAttemptId: 'at-story-033',
  },
  {
    workerId: 'worker-north',
    status: 'idle',
    heartbeatStats: { queueDepth: 0 },
    lastHeartbeatTime: now - 90,
    lastDequeueTime: now - 3600,
    lastBusyTime: now - 5400,
    lastIdleTime: now - 5400,
    currentRolloutId: null,
    currentAttemptId: null,
  },
  {
    workerId: 'worker-south',
    status: 'idle',
    heartbeatStats: null,
    lastHeartbeatTime: now - 900,
    lastDequeueTime: now - 7200,
    lastBusyTime: now - 8600,
    lastIdleTime: now - 8600,
    currentRolloutId: null,
    currentAttemptId: null,
  },
  {
    workerId: 'worker-standby',
    status: 'unknown',
    heartbeatStats: { queueDepth: 0 },
    lastHeartbeatTime: now - 15,
    lastDequeueTime: now - 4000,
    lastBusyTime: null,
    lastIdleTime: null,
    currentRolloutId: null,
    currentAttemptId: null,
  },
];

type WorkersTableStoryWrapperProps = {
  maxWidth: number;
  initialSort?: { column: string; direction: 'asc' | 'desc' };
};

function WorkersTableStoryWrapper({ maxWidth, initialSort }: WorkersTableStoryWrapperProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [recordsPerPage, setRecordsPerPage] = useState(5);
  const [sort, setSort] = useState<{ column: string; direction: 'asc' | 'desc' }>(
    () => initialSort ?? { column: 'lastHeartbeatTime', direction: 'desc' },
  );

  const filteredWorkers = useMemo(() => {
    const normalized = searchTerm.trim().toLowerCase();
    if (!normalized) {
      return sampleWorkers;
    }
    return sampleWorkers.filter((worker) => worker.workerId.toLowerCase().includes(normalized));
  }, [searchTerm]);

  return (
    <Stack gap='md' p='lg'>
      <Title order={2}>Workers ({maxWidth}px max width)</Title>
      <TextInput
        placeholder='Search'
        leftSection={<IconSearch size={16} />}
        value={searchTerm}
        onChange={(event) => setSearchTerm(event.currentTarget.value)}
        w='100%'
        style={{ maxWidth: 360 }}
      />
      <Box style={{ maxWidth }}>
        <WorkersTable
          workers={filteredWorkers}
          totalRecords={filteredWorkers.length}
          isFetching={false}
          isError={false}
          error={null}
          searchTerm={searchTerm}
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onSortStatusChange={(status) => {
            return setSort({ column: status.columnAccessor as string, direction: status.direction });
          }}
          onPageChange={setPage}
          onRecordsPerPageChange={setRecordsPerPage}
          onResetFilters={() => {
            setSearchTerm('');
            setPage(1);
          }}
          onRefetch={() => {}}
          onShowDetails={() => {}}
        />
      </Box>
    </Stack>
  );
}

export const Wide: Story = {
  render: () => <WorkersTableStoryWrapper maxWidth={1600} />,
};

export const Narrow: Story = {
  render: () => <WorkersTableStoryWrapper maxWidth={780} />,
};

export const SortedByCurrentRollout: Story = {
  render: () => (
    <WorkersTableStoryWrapper maxWidth={1200} initialSort={{ column: 'currentRolloutId', direction: 'asc' }} />
  ),
};
