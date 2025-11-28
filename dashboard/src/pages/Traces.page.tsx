// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect, useMemo, useState } from 'react';
import { skipToken } from '@reduxjs/toolkit/query';
import { IconCheck, IconChevronDown, IconSearch } from '@tabler/icons-react';
import type { DataTableSortStatus } from 'mantine-datatable';
import { useSearchParams } from 'react-router-dom';
import { Button, Group, Menu, Select, Skeleton, Stack, TextInput, Title } from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { TracesTable, type TracesTableRecord } from '@/components/TracesTable.component';
import { selectAutoRefreshMs } from '@/features/config';
import {
  useGetRolloutAttemptsQuery,
  useGetRolloutsQuery,
  useGetSpansQuery,
  type GetRolloutsQueryArgs,
} from '@/features/rollouts';
import {
  hydrateTracesStateFromQuery,
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
import { getErrorDescriptor } from '@/utils/error';
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

function mergeRolloutCache(cache: Record<string, Rollout>, items: Rollout[]): Record<string, Rollout> {
  if (!items.length) {
    return cache;
  }
  let changed = false;
  const next = { ...cache };
  for (const item of items) {
    const existing = next[item.rolloutId];
    if (!existing || existing !== item) {
      next[item.rolloutId] = item;
      changed = true;
    }
  }
  return changed ? next : cache;
}

export function TracesPage() {
  const dispatch = useAppDispatch();
  const autoRefreshMs = useAppSelector(selectAutoRefreshMs);
  const [searchParams, setSearchParams] = useSearchParams();
  const searchParamsKey = searchParams.toString();
  const [hydratedSearchParamsKey, setHydratedSearchParamsKey] = useState<string | null>(null);
  const [rolloutSearchValue, setRolloutSearchValue] = useState('');
  const [debouncedRolloutSearchValue] = useDebouncedValue(rolloutSearchValue, 300);
  const normalizedRolloutSearchValue = debouncedRolloutSearchValue.trim();
  const rolloutSearchActive = normalizedRolloutSearchValue.length > 0;
  const [rolloutLookup, setRolloutLookup] = useState<Record<string, Rollout>>({});
  const hasRolloutQueryParam = searchParams.has('rolloutId');
  const hasAttemptQueryParam = searchParams.has('attemptId');
  const [initialHasRolloutQueryParam] = useState(hasRolloutQueryParam);
  const rolloutIdFromQuery = hasRolloutQueryParam ? searchParams.get('rolloutId') || null : undefined;
  const attemptIdFromQuery = hasAttemptQueryParam ? searchParams.get('attemptId') || null : undefined;
  const rolloutId = useAppSelector(selectTracesRolloutId);
  const attemptId = useAppSelector(selectTracesAttemptId);
  const searchTerm = useAppSelector(selectTracesSearchTerm);
  const page = useAppSelector(selectTracesPage);
  const recordsPerPage = useAppSelector(selectTracesRecordsPerPage);
  const sort = useAppSelector(selectTracesSort);
  const viewMode = useAppSelector(selectTracesViewMode);
  const spansQueryArgs = useAppSelector(selectTracesQueryArgs);

  const baseRolloutsQueryArgs = useMemo<GetRolloutsQueryArgs>(
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
  } = useGetRolloutsQuery(baseRolloutsQueryArgs, {
    pollingInterval: autoRefreshMs > 0 ? autoRefreshMs : undefined,
  });

  const baseRolloutItems = rolloutsData?.items ?? [];

  const rolloutSearchQueryArgs = useMemo<GetRolloutsQueryArgs | null>(() => {
    if (!rolloutSearchActive) {
      return null;
    }
    return {
      limit: 20,
      offset: 0,
      sortBy: 'start_time',
      sortOrder: 'desc',
      rolloutIdContains: normalizedRolloutSearchValue,
    };
  }, [normalizedRolloutSearchValue, rolloutSearchActive]);

  const { data: rolloutSearchData, isFetching: rolloutSearchFetching } = useGetRolloutsQuery(
    rolloutSearchQueryArgs ?? skipToken,
  );

  const rolloutByIdQueryArgs = useMemo<GetRolloutsQueryArgs | null>(() => {
    if (!rolloutId || rolloutLookup[rolloutId]) {
      return null;
    }
    return {
      limit: 20,
      offset: 0,
      sortBy: 'start_time',
      sortOrder: 'desc',
      rolloutIdContains: rolloutId,
    };
  }, [rolloutId, rolloutLookup]);

  const { data: rolloutByIdData, isFetching: rolloutByIdFetching } = useGetRolloutsQuery(
    rolloutByIdQueryArgs ?? skipToken,
  );

  const searchRolloutItems = rolloutSearchData?.items ?? [];
  const rolloutByIdItems = rolloutByIdData?.items ?? [];

  useEffect(() => {
    if (baseRolloutItems.length > 0) {
      setRolloutLookup((prev) => mergeRolloutCache(prev, baseRolloutItems));
    }
  }, [baseRolloutItems]);

  useEffect(() => {
    if (searchRolloutItems.length > 0) {
      setRolloutLookup((prev) => mergeRolloutCache(prev, searchRolloutItems));
    }
  }, [searchRolloutItems]);

  useEffect(() => {
    if (rolloutByIdItems.length > 0) {
      setRolloutLookup((prev) => mergeRolloutCache(prev, rolloutByIdItems));
    }
  }, [rolloutByIdItems]);

  const selectedRollout = rolloutId ? (rolloutLookup[rolloutId] ?? null) : null;

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

  const shouldResolveRollout = rolloutId !== null && !rolloutLookup[rolloutId];

  useEffect(() => {
    if (!rolloutsData) {
      return;
    }

    if (rolloutsData.total === 0) {
      if (rolloutId !== null) {
        dispatch(setTracesRolloutId(null));
      }
      return;
    }

    if (rolloutId === null) {
      if (!initialHasRolloutQueryParam && baseRolloutItems[0]) {
        dispatch(setTracesRolloutId(baseRolloutItems[0].rolloutId));
      }
      return;
    }

    if (!shouldResolveRollout) {
      return;
    }

    if (rolloutSearchActive && normalizedRolloutSearchValue === rolloutId) {
      return;
    }

    if (rolloutByIdQueryArgs) {
      if (rolloutByIdFetching || !rolloutByIdData || rolloutsLoading) {
        return;
      }
      if (rolloutByIdData.items.length === 0) {
        dispatch(setTracesRolloutId(null));
      }
      return;
    }

    dispatch(setTracesRolloutId(null));
  }, [
    baseRolloutItems,
    dispatch,
    initialHasRolloutQueryParam,
    normalizedRolloutSearchValue,
    rolloutByIdData,
    rolloutByIdFetching,
    rolloutByIdQueryArgs,
    rolloutId,
    rolloutLookup,
    rolloutSearchActive,
    rolloutsData,
    rolloutsLoading,
    shouldResolveRollout,
  ]);

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

    if (attemptId === null) {
      const fallbackAttemptId = selectedRollout?.attempt?.attemptId ?? null;
      if (fallbackAttemptId !== attemptId) {
        dispatch(setTracesAttemptId(fallbackAttemptId));
      }
    }
  }, [attemptsData, attemptId, dispatch, rolloutId, selectedRollout]);

  const visibleRolloutItems = rolloutSearchActive ? searchRolloutItems : baseRolloutItems;
  const rolloutSelectIsFetching =
    (rolloutSearchActive ? rolloutSearchFetching : rolloutsFetching) ||
    Boolean(rolloutByIdQueryArgs && rolloutByIdFetching);

  const rolloutOptions = useMemo(() => {
    const options = visibleRolloutItems.map((rollout) => ({
      value: rollout.rolloutId,
      label: rollout.rolloutId,
    }));
    if (rolloutId && !visibleRolloutItems.some((rollout) => rollout.rolloutId === rolloutId)) {
      options.push({ value: rolloutId, label: rolloutId });
    }
    return options;
  }, [rolloutId, visibleRolloutItems]);

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

  const attemptPlaceholder = useMemo(() => {
    if (!rolloutId) {
      return 'Select Attempt';
    }
    if (attemptOptions.length === 0) {
      return 'No Attempt';
    }
    return 'Latest Attempt';
  }, [attemptOptions.length, rolloutId]);

  const rawSpansData = spansData as any as { items?: Span[]; total?: number } | undefined;
  const spans = rawSpansData?.items ?? [];
  const spansTotal = rawSpansData?.total ?? 0;
  const recordsPerPageOptions = [50, 100, 200, 500];
  const isInitialLoading = rolloutsLoading && baseRolloutItems.length === 0;
  const isFetching = spansFetching || rolloutsFetching || attemptsFetching;

  const selectionMessage = useMemo<string | undefined>(() => {
    if (!rolloutId && !attemptId) {
      return 'Select a rollout and attempt to view traces.';
    }
    if (!rolloutId) {
      return 'Select a rollout to view traces.';
    }
    if (!attemptId) {
      return 'Select an attempt to view traces.';
    }
    return undefined;
  }, [attemptId, rolloutId]);

  const tableSpans = selectionMessage ? [] : spans;
  const tableIsError = selectionMessage ? false : spansIsError;
  const tableError = selectionMessage ? undefined : spansError;
  const tableIsFetching = selectionMessage ? false : isFetching;

  useEffect(() => {
    const anyError = rolloutsIsError || attemptsIsError || spansIsError;
    if (anyError) {
      const descriptor =
        (rolloutsIsError && getErrorDescriptor(rolloutsError)) ||
        (attemptsIsError && getErrorDescriptor(attemptsError)) ||
        (spansIsError && getErrorDescriptor(spansError)) ||
        null;
      const detailSuffix = descriptor ? ` (${descriptor})` : '';
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

  useEffect(() => {
    const payload: { rolloutId?: string | null; attemptId?: string | null } = {};
    if (hasRolloutQueryParam) {
      payload.rolloutId = rolloutIdFromQuery;
    }
    if (hasAttemptQueryParam) {
      payload.attemptId = attemptIdFromQuery;
    }
    if (Object.keys(payload).length > 0) {
      dispatch(hydrateTracesStateFromQuery(payload));
    }
    setHydratedSearchParamsKey((prev) => (prev === searchParamsKey ? prev : searchParamsKey));
  }, [attemptIdFromQuery, dispatch, hasAttemptQueryParam, hasRolloutQueryParam, rolloutIdFromQuery, searchParamsKey]);

  useEffect(() => {
    if (hydratedSearchParamsKey !== searchParamsKey) {
      return;
    }
    const next = new URLSearchParams(searchParams);
    let changed = false;

    if (rolloutId) {
      if (next.get('rolloutId') !== rolloutId) {
        next.set('rolloutId', rolloutId);
        changed = true;
      }
    } else if (next.has('rolloutId')) {
      next.delete('rolloutId');
      changed = true;
    }

    if (rolloutId && attemptId) {
      if (next.get('attemptId') !== attemptId) {
        next.set('attemptId', attemptId);
        changed = true;
      }
    } else if (next.has('attemptId')) {
      next.delete('attemptId');
      changed = true;
    }

    if (changed) {
      setSearchParams(next, { replace: true });
    }
  }, [attemptId, hydratedSearchParamsKey, rolloutId, searchParams, searchParamsKey, setSearchParams]);

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
      const rollout = rolloutLookup[record.rolloutId];
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
    [attemptsData, dispatch, rolloutLookup],
  );

  const handleShowSpanDetail = useCallback(
    (record: TracesTableRecord) => {
      const rolloutForSpan = rolloutLookup[record.rolloutId] ?? null;
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
    [attemptsData, dispatch, rolloutLookup],
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
                setRolloutSearchValue('');
              }}
              searchable
              searchValue={rolloutSearchValue}
              onSearchChange={(value) => {
                setRolloutSearchValue(value ?? '');
              }}
              placeholder='Select rollout'
              aria-label='Select rollout'
              nothingFoundMessage={
                rolloutSelectIsFetching ? 'Loading...' : rolloutSearchActive ? 'No matching rollouts' : 'No rollouts'
              }
              comboboxProps={{ withinPortal: true }}
              onDropdownOpen={() => {
                setRolloutSearchValue('');
              }}
              onDropdownClose={() => {
                setRolloutSearchValue('');
              }}
              w={260}
              disabled={rolloutOptions.length === 0 && !rolloutSelectIsFetching}
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
              placeholder={attemptPlaceholder}
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
          spans={tableSpans}
          totalRecords={selectionMessage ? 0 : spansTotal}
          isFetching={tableIsFetching}
          isError={tableIsError}
          error={tableError}
          selectionMessage={selectionMessage}
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
