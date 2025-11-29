// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Editor } from '@monaco-editor/react';
import { IconCheck, IconCopy } from '@tabler/icons-react';
import type { DataTableSortStatus } from 'mantine-datatable';
import { createSearchParams, Link, useInRouterContext, useLocation } from 'react-router-dom';
import {
  ActionIcon,
  Anchor,
  Badge,
  Box,
  CopyButton,
  Drawer,
  Group,
  Stack,
  Text,
  Tooltip,
  useMantineColorScheme,
} from '@mantine/core';
import { useGetSpansQuery } from '@/features/rollouts';
import { closeDrawer, openDrawer, selectDrawerContent, selectDrawerIsOpen } from '@/features/ui/drawer';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import type { Attempt, AttemptStatus, Rollout, RolloutStatus, Span, Worker } from '@/types';
import { formatStatusLabel } from '@/utils/format';
import { TracesTable, type TracesTableRecord } from './TracesTable.component';

const ATTEMPT_STATUS_COLORS: Record<AttemptStatus, string> = {
  failed: 'red',
  preparing: 'violet',
  running: 'blue',
  succeeded: 'teal',
  timeout: 'orange',
  unresponsive: 'orange',
};

const ROLLOUT_STATUS_COLORS: Record<RolloutStatus, string> = {
  cancelled: 'gray',
  failed: 'red',
  preparing: 'violet',
  queuing: 'blue',
  requeuing: 'cyan',
  running: 'blue',
  succeeded: 'teal',
};

const SPAN_STATUS_COLORS: Record<Span['status']['status_code'], string> = {
  UNSET: 'gray',
  OK: 'teal',
  ERROR: 'red',
};

const WORKER_STATUS_COLORS: Record<Worker['status'], string> = {
  busy: 'orange',
  idle: 'teal',
  unknown: 'gray',
};

const TRACES_SORT_FIELD_MAP: Record<string, string> = {
  name: 'name',
  traceId: 'trace_id',
  spanId: 'span_id',
  parentId: 'parent_id',
  statusCode: 'status_code',
  startTime: 'start_time',
  duration: 'duration',
};

type SortDirection = 'asc' | 'desc';

type LocalSortState = {
  column: string;
  direction: SortDirection;
};

function resolveTracesSortField(column: string): string {
  return TRACES_SORT_FIELD_MAP[column] ?? 'start_time';
}

function getStatusBadgeColor(status: RolloutStatus | AttemptStatus, isAttempt: boolean) {
  if (isAttempt) {
    return ATTEMPT_STATUS_COLORS[status as AttemptStatus] ?? 'gray';
  }

  return ROLLOUT_STATUS_COLORS[status as RolloutStatus] ?? 'gray';
}

function formatJson(value: unknown) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export type AppDrawerProps = {
  opened: boolean;
  onClose: () => void;
  title?: ReactNode;
  body?: ReactNode;
};

export function AppDrawer({ opened, onClose, title, body }: AppDrawerProps) {
  return (
    <Drawer
      position='right'
      size='lg'
      opened={opened}
      onClose={onClose}
      overlayProps={{ opacity: 0.5 }}
      withinPortal
      styles={{
        content: {
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '100vh',
        },
        body: {
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          padding: 'var(--mantine-spacing-md)',
          minHeight: 0,
          overflow: 'hidden',
        },
      }}
      title={title}
    >
      <Stack gap='md' h='100%' style={{ flex: 1, minHeight: 0 }}>
        {body}
      </Stack>
    </Drawer>
  );
}

type TraceDrawerTitleProps = {
  span: Span;
};

export function TraceDrawerTitle({ span }: TraceDrawerTitleProps) {
  const spanStatusCode = span.status?.status_code ?? null;
  const spanBadgeColor = spanStatusCode ? (SPAN_STATUS_COLORS[spanStatusCode] ?? 'gray') : undefined;

  return (
    <Stack gap={3}>
      <Group gap={6}>
        <Text fw={600}>{span.name ?? span.spanId}</Text>
        {spanStatusCode ? (
          <Badge size='sm' variant='light' color={spanBadgeColor}>
            {spanStatusCode}
          </Badge>
        ) : null}
      </Group>
      <Group gap={6}>
        <Text size='sm' c='dimmed'>
          {span.spanId}
        </Text>
        <CopyButton value={span.spanId}>
          {({ copied, copy }) => (
            <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
              <ActionIcon
                aria-label={`Copy span ID ${span.spanId}`}
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
      <Group gap='xs'>
        <Group gap={3}>
          <Text size='sm' c='dimmed' fw={500}>
            Rollout
          </Text>
          <Text size='sm' c='dimmed'>
            {span.rolloutId}
          </Text>
        </Group>
        <Group gap={3}>
          <Text size='sm' c='dimmed' fw={500}>
            Attempt
          </Text>
          <Text size='sm' c='dimmed'>
            {span.attemptId ?? '—'}
          </Text>
        </Group>
      </Group>
    </Stack>
  );
}

type RolloutAttemptDrawerTitleProps = {
  rollout: Rollout;
  attempt: Attempt | null;
};

export function RolloutAttemptDrawerTitle({ rollout, attempt }: RolloutAttemptDrawerTitleProps) {
  const rolloutId = rollout.rolloutId;
  const attemptId = attempt?.attemptId ?? null;
  const rolloutStatus = rollout.status ?? null;
  const attemptStatus = attempt?.status ?? null;
  const rolloutStatusLabel = rolloutStatus ? formatStatusLabel(rolloutStatus) : null;
  const attemptStatusLabel = attemptStatus ? formatStatusLabel(attemptStatus) : null;
  const hasStatusMismatch = rolloutStatus !== null && attemptStatus !== null && rolloutStatus !== attemptStatus;
  const rolloutBadgeColor = rolloutStatus ? getStatusBadgeColor(rolloutStatus, false) : undefined;
  const attemptBadgeColor = attemptStatus ? getStatusBadgeColor(attemptStatus, true) : undefined;
  const showRolloutBadgeInHeading = Boolean(rolloutStatusLabel && (!attemptStatus || hasStatusMismatch));
  const showAttemptBadge = Boolean(attemptStatusLabel && attemptStatus);

  return (
    <Stack gap={3}>
      <Group gap={6}>
        <Text fw={600}>{rolloutId}</Text>
        <CopyButton value={rolloutId}>
          {({ copied, copy }) => (
            <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
              <ActionIcon
                aria-label={`Copy rollout ID ${rolloutId}`}
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
        {showRolloutBadgeInHeading && rolloutStatusLabel ? (
          <Badge size='sm' variant='light' color={rolloutBadgeColor}>
            {rolloutStatusLabel}
          </Badge>
        ) : null}
      </Group>
      <Group gap='xs'>
        {attemptId ? (
          <Group gap={3}>
            <Text size='sm' c='dimmed' fw={500}>
              Attempt
            </Text>
            <Text size='sm' c='dimmed'>
              {attemptId}
            </Text>
          </Group>
        ) : null}
        {showAttemptBadge && attemptStatusLabel ? (
          <Badge size='sm' variant='light' color={attemptBadgeColor}>
            {attemptStatusLabel}
          </Badge>
        ) : null}
        {!showRolloutBadgeInHeading && !attemptStatus && rolloutStatusLabel ? (
          <Badge size='sm' variant='light' color={rolloutBadgeColor}>
            {rolloutStatusLabel}
          </Badge>
        ) : null}
      </Group>
    </Stack>
  );
}

type JsonEditorProps = {
  value: unknown;
};

export function JsonEditor({ value }: JsonEditorProps) {
  const { colorScheme } = useMantineColorScheme();
  const editorTheme = colorScheme === 'dark' ? 'vs-dark' : 'vs-light';

  return (
    <Box data-testid='json-editor-container' style={{ flex: 1, minHeight: 0 }}>
      <Editor
        height='100%'
        language='json'
        value={formatJson(value)}
        theme={editorTheme}
        options={{
          readOnly: true,
          domReadOnly: true,
          minimap: { enabled: false },
          automaticLayout: true,
          scrollBeyondLastLine: false,
          fontSize: 13,
        }}
      />
    </Box>
  );
}

type RolloutTracesDrawerBodyProps = {
  rollout: Rollout;
  attempt: Attempt | null;
  onShowRollout: (record: TracesTableRecord) => void;
  onShowSpanDetail: (record: TracesTableRecord) => void;
};

function RolloutTracesDrawerBody({ rollout, attempt, onShowRollout, onShowSpanDetail }: RolloutTracesDrawerBodyProps) {
  const [page, setPage] = useState(1);
  const [recordsPerPage, setRecordsPerPage] = useState(100);
  const [sort, setSort] = useState<LocalSortState>({
    column: 'startTime',
    direction: 'desc',
  });

  useEffect(() => {
    setPage(1);
  }, [rollout.rolloutId, attempt?.attemptId]);

  const queryArgs = useMemo(
    () => ({
      rolloutId: rollout.rolloutId,
      attemptId: attempt?.attemptId ?? undefined,
      limit: recordsPerPage,
      offset: Math.max(0, (page - 1) * recordsPerPage),
      sortBy: resolveTracesSortField(sort.column),
      sortOrder: sort.direction,
    }),
    [rollout.rolloutId, attempt?.attemptId, recordsPerPage, page, sort],
  );

  const { data, isFetching, isError, error, refetch } = useGetSpansQuery(queryArgs);
  const spans = data?.items ?? [];
  const totalRecords = data?.total ?? 0;
  const tracesLinkSearch = useMemo(() => {
    const params = createSearchParams({
      rolloutId: rollout.rolloutId,
      ...(attempt?.attemptId ? { attemptId: attempt.attemptId } : {}),
    });
    return params.toString();
  }, [attempt?.attemptId, rollout.rolloutId]);
  const tracesLinkHref = tracesLinkSearch ? `/traces?${tracesLinkSearch}` : '/traces';
  const isWithinRouter = useInRouterContext();

  const handleSortStatusChange = useCallback((status: DataTableSortStatus<TracesTableRecord>) => {
    setSort({
      column: status.columnAccessor as string,
      direction: status.direction,
    });
  }, []);

  const handlePageChange = useCallback((nextPage: number) => {
    setPage(nextPage);
  }, []);

  const handleRecordsPerPageChange = useCallback((value: number) => {
    setRecordsPerPage(value);
    setPage(1);
  }, []);

  return (
    <Stack gap='md' style={{ flex: 1, minHeight: 0 }}>
      <Group justify='space-between' align='center' gap='sm' wrap='nowrap'>
        <Text size='sm' style={{ flex: 1, minWidth: 0 }}>
          Showing spans for{' '}
          <Text component='span' fw={600}>
            {rollout.rolloutId}
            {attempt ? ` · Attempt ${attempt.sequenceId} (${attempt.attemptId})` : ' · Latest attempt'}
          </Text>
        </Text>
        {isWithinRouter ? (
          <Anchor
            component={Link}
            to={tracesLinkHref}
            size='sm'
            aria-label={`Open traces page for rollout ${rollout.rolloutId}${
              attempt ? ` attempt ${attempt.sequenceId}` : ''
            }`}
          >
            View full traces
          </Anchor>
        ) : (
          <Anchor
            href={tracesLinkHref}
            size='sm'
            aria-label={`Open traces page for rollout ${rollout.rolloutId}${
              attempt ? ` attempt ${attempt.sequenceId}` : ''
            }`}
          >
            View full traces
          </Anchor>
        )}
      </Group>
      <Box data-testid='traces-drawer-table-container' style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <TracesTable
          spans={spans}
          totalRecords={totalRecords}
          isFetching={isFetching}
          isError={isError}
          error={error}
          searchTerm=''
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onSortStatusChange={handleSortStatusChange}
          onPageChange={handlePageChange}
          onRecordsPerPageChange={handleRecordsPerPageChange}
          onResetFilters={() => {}}
          onRefetch={refetch}
          onShowRollout={onShowRollout}
          onShowSpanDetail={onShowSpanDetail}
          recordsPerPageOptions={[50, 100, 200, 500]}
        />
      </Box>
    </Stack>
  );
}

type WorkerDrawerTitleProps = {
  worker: Worker;
};

function WorkerDrawerTitle({ worker }: WorkerDrawerTitleProps) {
  const badgeColor = WORKER_STATUS_COLORS[worker.status] ?? 'gray';
  return (
    <Stack gap={3}>
      <Group gap={6} align='center'>
        <Text fw={600}>{worker.workerId}</Text>
        <CopyButton value={worker.workerId}>
          {({ copied, copy }) => (
            <Tooltip label={copied ? 'Copied' : 'Copy'} withArrow>
              <ActionIcon
                aria-label={`Copy worker ID ${worker.workerId}`}
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
        <Badge size='sm' variant='light' color={badgeColor}>
          {formatStatusLabel(worker.status)}
        </Badge>
      </Group>
      <Group gap='xl'>
        <Group gap={4}>
          <Text size='sm' c='dimmed' fw={500}>
            Rollout
          </Text>
          <Text size='sm' c='dimmed'>
            {worker.currentRolloutId ?? '—'}
          </Text>
        </Group>
        <Group gap={4}>
          <Text size='sm' c='dimmed' fw={500}>
            Attempt
          </Text>
          <Text size='sm' c='dimmed'>
            {worker.currentAttemptId ?? '—'}
          </Text>
        </Group>
      </Group>
    </Stack>
  );
}

export function AppDrawerContainer() {
  const dispatch = useAppDispatch();
  const isOpen = useAppSelector(selectDrawerIsOpen);
  const content = useAppSelector(selectDrawerContent);
  const isRouterAvailable = useInRouterContext();

  const handleClose = useCallback(() => {
    dispatch(closeDrawer());
  }, [dispatch]);
  const handleNavigation = useCallback(() => {
    if (isOpen) {
      dispatch(closeDrawer());
    }
  }, [dispatch, isOpen]);

  const derivedContent = useMemo(() => {
    if (!content) {
      return null;
    }

    if (content.type === 'worker-detail') {
      const { worker } = content;
      const title = <WorkerDrawerTitle worker={worker} />;
      const body = <JsonEditor value={worker} />;
      return { title, body };
    }

    if (content.type === 'trace-detail') {
      const { span } = content;
      const title = <TraceDrawerTitle span={span} />;
      const body = <JsonEditor value={span} />;

      return { title, body };
    }

    const rollout = content.rollout;
    const attempt = content.attempt;
    const title = <RolloutAttemptDrawerTitle rollout={rollout} attempt={attempt} />;

    if (content.type === 'rollout-json') {
      const jsonValue = content.isNested && content.attempt ? content.attempt : rollout;
      const body = jsonValue ? <JsonEditor value={jsonValue} /> : null;
      return { title, body };
    }

    if (content.type === 'rollout-traces') {
      const body = (
        <RolloutTracesDrawerBody
          rollout={rollout}
          attempt={attempt}
          onShowRollout={() => {
            const attemptForRecord = attempt ?? rollout.attempt ?? null;
            dispatch(
              openDrawer({
                type: 'rollout-json',
                rollout,
                attempt: attemptForRecord,
                isNested: content.isNested,
              }),
            );
          }}
          onShowSpanDetail={(record) => {
            const attemptForRecord = attempt ?? rollout.attempt ?? null;
            dispatch(
              openDrawer({
                type: 'trace-detail',
                span: record,
                rollout,
                attempt: attemptForRecord,
              }),
            );
          }}
        />
      );

      return { title, body };
    }

    return null;
  }, [content, dispatch]);

  if (!content || !derivedContent) {
    return null;
  }

  const { title, body } = derivedContent;

  return (
    <>
      {isRouterAvailable ? <DrawerLocationWatcher onNavigation={handleNavigation} /> : null}
      <AppDrawer opened={isOpen} onClose={handleClose} title={title} body={body} />
    </>
  );
}

type DrawerLocationWatcherProps = {
  onNavigation: () => void;
};

function DrawerLocationWatcher({ onNavigation }: DrawerLocationWatcherProps) {
  const location = useLocation();
  const lastLocationKeyRef = useRef(location.key);

  useEffect(() => {
    if (lastLocationKeyRef.current === location.key) {
      return;
    }
    lastLocationKeyRef.current = location.key;
    onNavigation();
  }, [location.key, onNavigation]);

  return null;
}
