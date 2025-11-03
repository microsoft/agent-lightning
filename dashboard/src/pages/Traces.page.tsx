// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect, useMemo } from 'react';
import { skipToken } from '@reduxjs/toolkit/query';
import { IconCheck, IconChevronDown, IconSearch } from '@tabler/icons-react';
import type { DataTableSortStatus } from 'mantine-datatable';
import { Button, Group, Menu, Select, Skeleton, Stack, TextInput, Title } from '@mantine/core';
import { TracesTable, type TracesTableRecord } from '@/components/TracesTable.component';
import { selectAutoRefreshMs } from '@/features/config';
import {
  useGetRolloutAttemptsQuery,
  useGetRolloutsQuery,
  useGetSpansQuery,
  type GetRolloutsQueryArgs,
} from '@/features/rollouts';
import {
  resetTracesFilters,
  selectTracesAttemptId,
  selectTracesPage,
  selectTracesQueryArgs,
  selectTracesRecordsPerPage,
  selectTracesRolloutId,
  selectTracesSearchTerm,
  selectTracesSort,
  selectTracesViewMode,
  setTracesAttemptId,
  setTracesPage,
  setTracesRecordsPerPage,
  setTracesRolloutId,
  setTracesSearchTerm,
  setTracesSort,
  setTracesViewMode,
} from '@/features/traces';
import { hideAlert, showAlert } from '@/features/ui/alert';
import { openDrawer } from '@/features/ui/drawer';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import type { Attempt, Rollout, Span } from '@/types';
import { formatStatusLabel } from '@/utils/format';

const VIEW_OPTIONS = [
  { value: 'table', label: 'Table View', disabled: false },
  { value: 'waterfall', label: 'Waterfall View (Coming Soon)', disabled: true },
  { value: 'tree', label: 'Tree View (Coming Soon)', disabled: true },
] as const;

type ViewOptionValue = (typeof VIEW_OPTIONS)[number]['value'];

function getLatestAttempt(attempts: Attempt[]): Attempt | null {
  if (!attempts.length) {
    return null;
  }
  return [...attempts].sort((a, b) => a.sequenceId - b.sequenceId).at(-1) ?? null;
}

function findRollout(rollouts: Rollout[] | undefined, rolloutId: string | null): Rollout | null {
  if (!rollouts || !rolloutId) {
    return null;
  }
  return rollouts.find((rollout) => rollout.rolloutId === rolloutId) ?? null;
}

export function TracesPage() {
  const dispatch = useAppDispatch();
  const autoRefreshMs = useAppSelector(selectAutoRefreshMs);
  const rolloutId = useAppSelector(selectTracesRolloutId);
  const attemptId = useAppSelector(selectTracesAttemptId);
  const searchTerm = useAppSelector(selectTracesSearchTerm);
  const page = useAppSelector(selectTracesPage);
  const recordsPerPage = useAppSelector(selectTracesRecordsPerPage);
  const sort = useAppSelector(selectTracesSort);
  const viewMode = useAppSelector(selectTracesViewMode);
  const spansQueryArgs = useAppSelector(selectTracesQueryArgs);

  const rolloutsQueryArgs = useMemo<GetRolloutsQueryArgs>(
    () => ({
      limit: 100,
      offset: 0,
      sortBy: 'start_time',
      sortOrder: 'desc',
    }),
    [],
  );

  const {
    data: rolloutsData,
    isLoading: rolloutsLoading,
    isFetching: rolloutsFetching,
    isError: rolloutsIsError,
    error: rolloutsError,
  } = useGetRolloutsQuery(rolloutsQueryArgs, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  const rolloutItems = rolloutsData?.items ?? [];

  const selectedRollout = useMemo(() => findRollout(rolloutItems, rolloutId), [rolloutItems, rolloutId]);

  const attemptsQueryArgs =
    rolloutId !== null
      ? {
          rolloutId,
          limit: 200,
          sortBy: 'sequence_id',
          sortOrder: 'desc' as const,
        }
      : skipToken;

  const {
    data: attemptsData,
    isFetching: attemptsFetching,
    isError: attemptsIsError,
    error: attemptsError,
  } = useGetRolloutAttemptsQuery(attemptsQueryArgs, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  const {
    data: spansData,
    isFetching: spansFetching,
    isError: spansIsError,
    error: spansError,
    refetch: refetchSpans,
  } = useGetSpansQuery(spansQueryArgs ?? skipToken, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  useEffect(() => {
    if (!rolloutsData) {
      return;
    }
    if (rolloutItems.length === 0) {
      if (rolloutId !== null) {
        dispatch(setTracesRolloutId(null));
      }
      return;
    }
    const rolloutExists = rolloutId ? rolloutItems.some((rollout) => rollout.rolloutId === rolloutId) : false;
    if (!rolloutExists) {
      dispatch(setTracesRolloutId(rolloutItems[0].rolloutId));
    }
  }, [dispatch, rolloutsData, rolloutId, rolloutItems]);

  useEffect(() => {
    if (!rolloutId) {
      if (attemptId !== null) {
        dispatch(setTracesAttemptId(null));
      }
      return;
    }

    if (attemptsData && attemptsData.items.length > 0) {
      const hasSelected = attemptId ? attemptsData.items.some((attempt) => attempt.attemptId === attemptId) : false;
      if (!hasSelected) {
        const latest = getLatestAttempt(attemptsData.items);
        if (latest && latest.attemptId !== attemptId) {
          dispatch(setTracesAttemptId(latest.attemptId));
        }
      }
      return;
    }

    const fallbackAttemptId = selectedRollout?.attempt?.attemptId ?? null;
    if (fallbackAttemptId !== attemptId) {
      dispatch(setTracesAttemptId(fallbackAttemptId));
    }
  }, [attemptsData, attemptId, dispatch, rolloutId, selectedRollout]);

  const rolloutOptions = useMemo(
    () =>
      rolloutItems.map((rollout) => ({
        value: rollout.rolloutId,
        label: rollout.rolloutId,
      })),
    [rolloutItems],
  );

  const attemptOptions = useMemo(() => {
    if (attemptsData && attemptsData.items.length > 0) {
      return [...attemptsData.items]
        .sort((a, b) => b.sequenceId - a.sequenceId)
        .map((attempt) => ({
          value: attempt.attemptId,
          label: `Attempt ${attempt.sequenceId} (${attempt.attemptId}) - ${formatStatusLabel(attempt.status)}`,
        }));
    }
    if (selectedRollout?.attempt) {
      const attempt = selectedRollout.attempt;
      return [
        {
          value: attempt.attemptId,
          label: `Attempt ${attempt.sequenceId} (${attempt.attemptId}) - ${formatStatusLabel(attempt.status)}`,
        },
      ];
    }
    return [];
  }, [attemptsData, selectedRollout]);

  const rawSpansData = spansData as any as { items?: Span[]; total?: number } | undefined;
  const spans = rawSpansData?.items ?? [];
  const spansTotal = rawSpansData?.total ?? 0;
  const recordsPerPageOptions = [50, 100, 200, 500];
  const isInitialLoading = rolloutsLoading && rolloutItems.length === 0;
  const isFetching = spansFetching || rolloutsFetching || attemptsFetching;

  useEffect(() => {
    const anyError = rolloutsIsError || attemptsIsError || spansIsError;
    const extractStatus = (value: unknown): string | null => {
      if (value && typeof value === 'object' && 'status' in (value as Record<string, unknown>)) {
        return String((value as Record<string, unknown>).status);
      }
      return null;
    };
    const extractMessage = (value: unknown): string | null => {
      if (value && typeof value === 'object' && 'message' in (value as Record<string, unknown>)) {
        return String((value as Record<string, unknown>).message);
      }
      return null;
    };

    if (anyError) {
      const statusDetail =
        extractStatus(rolloutsError) ?? extractStatus(attemptsError) ?? extractStatus(spansError) ?? null;
      const messageDetail =
        extractMessage(rolloutsError) ?? extractMessage(attemptsError) ?? extractMessage(spansError) ?? null;
      const detailSuffix = statusDetail ? ` (status: ${statusDetail})` : messageDetail ? ` (${messageDetail})` : '';
      dispatch(
        showAlert({
          id: 'traces-fetch',
          message: `Unable to refresh traces${detailSuffix}. The table may be out of date until the connection recovers.`,
          tone: 'error',
        }),
      );
      return;
    }

    if (!rolloutsLoading && !rolloutsFetching && !spansFetching && !attemptsFetching) {
      dispatch(hideAlert({ id: 'traces-fetch' }));
    }
  }, [
    attemptsError,
    attemptsFetching,
    attemptsIsError,
    dispatch,
    rolloutsError,
    rolloutsFetching,
    rolloutsIsError,
    rolloutsLoading,
    spansError,
    spansFetching,
    spansIsError,
  ]);

  useEffect(
    () => () => {
      dispatch(hideAlert({ id: 'traces-fetch' }));
    },
    [dispatch],
  );

  const handleSearchTermChange = useCallback(
    (value: string) => {
      dispatch(setTracesSearchTerm(value));
    },
    [dispatch],
  );

  const handleSortStatusChange = useCallback(
    (status: DataTableSortStatus<TracesTableRecord>) => {
      dispatch(
        setTracesSort({
          column: status.columnAccessor as string,
          direction: status.direction,
        }),
      );
    },
    [dispatch],
  );

  const handlePageChange = useCallback(
    (nextPage: number) => {
      dispatch(setTracesPage(nextPage));
    },
    [dispatch],
  );

  const handleRecordsPerPageChange = useCallback(
    (value: number) => {
      dispatch(setTracesRecordsPerPage(value));
    },
    [dispatch],
  );

  const handleResetFilters = useCallback(() => {
    dispatch(resetTracesFilters());
  }, [dispatch]);

  const handleRefetch = useCallback(() => {
    if (rolloutId) {
      void refetchSpans();
    }
  }, [refetchSpans, rolloutId]);

  const handleShowRollout = useCallback(
    (record: TracesTableRecord) => {
      if (rolloutItems.length === 0) {
        return;
      }
      const rollout = rolloutItems.find((item) => item.rolloutId === record.rolloutId);
      if (!rollout) {
        return;
      }

      const attempts = attemptsData?.items ?? [];
      const attemptForRecord =
        attempts.find((attempt) => attempt.attemptId === record.attemptId) ?? rollout.attempt ?? null;

      dispatch(
        openDrawer({
          type: 'rollout-json',
          rollout,
          attempt: attemptForRecord,
          isNested: false,
        }),
      );
    },
    [attemptsData, dispatch, rolloutItems],
  );

  const handleShowSpanDetail = useCallback(
    (record: TracesTableRecord) => {
      const rolloutForSpan =
        rolloutItems.length > 0 ? (rolloutItems.find((item) => item.rolloutId === record.rolloutId) ?? null) : null;
      const attempts = attemptsData?.items ?? [];
      const attemptForSpan =
        attempts.find((attempt) => attempt.attemptId === record.attemptId) ?? rolloutForSpan?.attempt ?? null;

      dispatch(
        openDrawer({
          type: 'trace-detail',
          span: record,
          rollout: rolloutForSpan,
          attempt: attemptForSpan,
        }),
      );
    },
    [attemptsData, dispatch, rolloutItems],
  );

  const handleParentIdClick = useCallback(
    (parentId: string) => {
      dispatch(setTracesSearchTerm(parentId));
    },
    [dispatch],
  );

  const handleViewChange = useCallback(
    (value: ViewOptionValue) => {
      dispatch(setTracesViewMode(value));
    },
    [dispatch],
  );

  const activeViewLabel = VIEW_OPTIONS.find((option) => option.value === viewMode)?.label ?? 'Table View';

  return (
    <Stack gap='md'>
      <Group justify='space-between' align='flex-start'>
        <Stack gap='sm' style={{ flex: 1, minWidth: 0 }}>
          <Title order={1}>Traces</Title>
          <Group gap='md' wrap='wrap'>
            <Select
              data={rolloutOptions}
              value={rolloutId ?? null}
              onChange={(value) => {
                if (value !== rolloutId) {
                  dispatch(setTracesRolloutId(value));
                }
              }}
              searchable
              placeholder='Select rollout'
              aria-label='Select rollout'
              nothingFoundMessage={rolloutsFetching ? 'Loading...' : 'No rollouts'}
              comboboxProps={{ withinPortal: true }}
              w={260}
              disabled={rolloutOptions.length === 0}
            />
            <Select
              data={attemptOptions}
              value={attemptId ?? null}
              onChange={(value) => {
                if (value !== attemptId) {
                  dispatch(setTracesAttemptId(value));
                }
              }}
              searchable
              placeholder='Latest attempt'
              aria-label='Select attempt'
              nothingFoundMessage={attemptsFetching ? 'Loading...' : 'No attempts'}
              comboboxProps={{ withinPortal: true }}
              w={280}
              disabled={!rolloutId || attemptOptions.length === 0}
            />
            <TextInput
              value={searchTerm}
              onChange={(event) => handleSearchTermChange(event.currentTarget.value)}
              placeholder='Search spans'
              aria-label='Search spans'
              leftSection={<IconSearch size={16} />}
              w={280}
            />
          </Group>
        </Stack>
        <Menu shadow='md' position='bottom-end' withinPortal>
          <Menu.Target>
            <Button variant='light' rightSection={<IconChevronDown size={16} />} aria-label='Change traces view'>
              {activeViewLabel}
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            {VIEW_OPTIONS.map((option) => (
              <Menu.Item
                key={option.value}
                disabled={option.disabled}
                leftSection={option.value === viewMode && !option.disabled ? <IconCheck size={14} /> : null}
                onClick={() => {
                  if (!option.disabled) {
                    handleViewChange(option.value);
                  }
                }}
              >
                {option.label}
              </Menu.Item>
            ))}
          </Menu.Dropdown>
        </Menu>
      </Group>

      <Skeleton visible={isInitialLoading} radius='md'>
        <TracesTable
          spans={rolloutId ? spans : []}
          totalRecords={spansTotal}
          isFetching={isFetching}
          isError={spansIsError}
          error={spansError}
          searchTerm={searchTerm}
          sort={sort}
          page={page}
          recordsPerPage={recordsPerPage}
          onSortStatusChange={handleSortStatusChange}
          onPageChange={handlePageChange}
          onRecordsPerPageChange={handleRecordsPerPageChange}
          onResetFilters={handleResetFilters}
          onRefetch={handleRefetch}
          onShowRollout={handleShowRollout}
          onShowSpanDetail={handleShowSpanDetail}
          onParentIdClick={handleParentIdClick}
          recordsPerPageOptions={recordsPerPageOptions}
        />
      </Skeleton>
    </Stack>
  );
}
