// Copyright (c) Microsoft. All rights reserved.

import { useMemo, useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { IconSearch } from '@tabler/icons-react';
import { Box, Stack, TextInput, Title } from '@mantine/core';
import type { Resources } from '@/types';
import { ResourcesTable } from './ResourcesTable.component';
import { ResourcesTree } from './ResourcesTree.component';

const meta: Meta<typeof ResourcesTable> = {
  title: 'Components/ResourcesTable',
  component: ResourcesTable,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof ResourcesTable>;

const sampleResources: Resources[] = [
  {
    resourcesId: 'rs-story-001',
    version: 1,
    createTime: 1710806400,
    updateTime: 1713412800,
    resources: {
      model: {
        name: 'gpt-4',
        version: '2024-01-01',
        temperature: 0.7,
        maxTokens: 2048,
        topP: 0.9,
      },
      database: {
        host: 'db.example.com',
        port: 5432,
        name: 'production',
        pool: {
          min: 2,
          max: 10,
          idle: 30000,
        },
      },
      cache: {
        type: 'redis',
        host: 'cache.example.com',
        port: 6379,
        ttl: 3600,
      },
    },
  },
  {
    resourcesId: 'rs-story-002',
    version: 2,
    createTime: 1712217600,
    updateTime: 1714823200,
    resources: {
      model: {
        name: 'claude-3-opus',
        version: '2024-02-01',
        temperature: 0.5,
        maxTokens: 4096,
      },
      storage: {
        type: 's3',
        bucket: 'training-data',
        region: 'us-east-1',
        credentials: {
          accessKeyId: 'AKIA***',
          encrypted: true,
        },
      },
      compute: {
        instances: [
          { id: 'i-001', type: 't3.large', zone: 'us-east-1a' },
          { id: 'i-002', type: 't3.large', zone: 'us-east-1b' },
        ],
        autoScaling: {
          min: 2,
          max: 10,
          targetCpu: 70,
        },
      },
    },
  },
  {
    resourcesId: 'rs-story-003',
    version: 3,
    createTime: 1709251200,
    updateTime: 1711856800,
    resources: {
      model: {
        name: 'gpt-3.5-turbo',
        version: '2023-12-01',
        temperature: 0.8,
        maxTokens: 1024,
      },
      monitoring: {
        enabled: true,
        interval: 60,
        metrics: ['cpu', 'memory', 'disk', 'network'],
        alerts: {
          email: 'ops@example.com',
          slack: '#alerts',
          pagerduty: true,
        },
      },
    },
  },
  {
    resourcesId: 'rs-story-004',
    version: 1,
    createTime: 1706745600,
    updateTime: 1709347200,
    resources: {
      apiKeys: {
        openai: 'sk-***',
        anthropic: 'sk-ant-***',
        replicate: 'r8-***',
      },
      rateLimits: {
        requestsPerMinute: 100,
        tokensPerDay: 1000000,
        concurrent: 5,
      },
    },
  },
  {
    resourcesId: 'rs-story-005',
    version: 1,
    createTime: 1704067200,
    updateTime: 1706668800,
    resources: {},
  },
];

type WrapperProps = {
  maxWidth: number;
  resourcesList?: Resources[] | undefined;
  isFetching?: boolean;
  isError?: boolean;
  error?: unknown;
};

function ResourcesTableStoryWrapper({
  maxWidth,
  resourcesList = sampleResources,
  isFetching = false,
  isError = false,
  error = null,
}: WrapperProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [recordsPerPage, setRecordsPerPage] = useState(5);
  const [sort, setSort] = useState<{ column: string; direction: 'asc' | 'desc' }>({
    column: 'resourcesId',
    direction: 'asc',
  });

  const baseResources = resourcesList ?? [];

  const filteredResources = useMemo(() => {
    const normalized = searchTerm.trim().toLowerCase();
    if (normalized.length === 0) {
      return baseResources;
    }
    return baseResources.filter((resource) => resource.resourcesId.toLowerCase().includes(normalized));
  }, [baseResources, searchTerm]);

  const sortedResources = useMemo(() => {
    const items = filteredResources.slice();
    const resolveSortValue = (resource: Resources, column: string) => {
      switch (column) {
        case 'version':
          return resource.version;
        case 'createTime':
          return resource.createTime;
        case 'updateTime':
          return resource.updateTime;
        case 'resourceCount':
          return Object.keys(resource.resources ?? {}).length;
        case 'resourcesId':
        default:
          return resource.resourcesId;
      }
    };

    items.sort((a, b) => {
      const aValue = resolveSortValue(a, sort.column);
      const bValue = resolveSortValue(b, sort.column);

      if (aValue === bValue) {
        return 0;
      }

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return aValue - bValue;
      }

      return String(aValue).localeCompare(String(bValue));
    });

    if (sort.direction === 'desc') {
      items.reverse();
    }

    return items;
  }, [filteredResources, sort]);

  const totalRecordsValue = sortedResources.length;

  const pagedResources = useMemo(() => {
    const startIndex = (page - 1) * recordsPerPage;
    return sortedResources.slice(startIndex, startIndex + recordsPerPage);
  }, [page, recordsPerPage, sortedResources]);

  return (
    <Box mx='auto' style={{ maxWidth, width: '100%', padding: 16 }}>
      <Stack gap='md'>
        <Title order={2}>Resources</Title>
        <TextInput
          placeholder='Search by Resources ID'
          value={searchTerm}
          onChange={(event) => {
            setSearchTerm(event.currentTarget.value);
            setPage(1);
          }}
          leftSection={<IconSearch size={16} />}
          data-testid='resources-search-input'
          w='100%'
          style={{ maxWidth: 360 }}
        />
        <ResourcesTable
          resourcesList={pagedResources}
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
            setSort({ column: 'resourcesId', direction: 'asc' });
            setPage(1);
          }}
          onRefetch={() => undefined}
          recordsPerPageOptions={[5, 10, 20]}
          renderRowExpansion={({ resources }) => <ResourcesTree resources={resources} />}
        />
      </Stack>
    </Box>
  );
}

export const WideContainer: Story = {
  render: () => <ResourcesTableStoryWrapper maxWidth={1280} />,
};

export const MediumContainer: Story = {
  render: () => <ResourcesTableStoryWrapper maxWidth={960} />,
};

export const NarrowContainer: Story = {
  render: () => <ResourcesTableStoryWrapper maxWidth={720} />,
};

export const DrawerWidth: Story = {
  render: () => <ResourcesTableStoryWrapper maxWidth={520} />,
};

export const ErrorState: Story = {
  render: () => (
    <ResourcesTableStoryWrapper maxWidth={600} resourcesList={[]} isError error={new Error('Network unreachable')} />
  ),
};

export const EmptyResources: Story = {
  render: () => (
    <ResourcesTableStoryWrapper
      maxWidth={960}
      resourcesList={[
        {
          resourcesId: 'rs-empty-001',
          version: 1,
          createTime: 1702000000,
          updateTime: 1704600000,
          resources: {},
        },
      ]}
    />
  ),
};

export const LoadingState: Story = {
  render: () => <ResourcesTableStoryWrapper maxWidth={960} resourcesList={[]} isFetching />,
};
