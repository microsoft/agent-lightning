// Copyright (c) Microsoft. All rights reserved.

import type { BaseQueryFn } from '@reduxjs/toolkit/query';
import { createApi, fetchBaseQuery, type FetchArgs, type FetchBaseQueryError } from '@reduxjs/toolkit/query/react';
import type { RootState } from '@/store';
import { camelCaseKeys } from '@/utils/format';
import type {
  Attempt,
  PaginatedResponse,
  Resources,
  Rollout,
  RolloutMode,
  RolloutStatus,
  Span,
  Timestamp,
  Worker,
  WorkerStatus,
} from '../../types';

const rawBaseQuery = fetchBaseQuery({
  baseUrl: '/',
});

const buildAbsoluteUrl = (baseUrl: string, path: string) => {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }

  const normalizedBase = baseUrl.replace(/\/+$/, '');
  const normalizedPath = path.replace(/^\/+/, '');
  if (!normalizedBase) {
    return `/${normalizedPath}`;
  }
  return `${normalizedBase}/${normalizedPath}`;
};

const normalizeHeartbeat = (
  attempt: Partial<Attempt> & { lastHeartbeatTime?: Timestamp | null; lastHeartBeatTime?: Timestamp | null },
): Timestamp | null => {
  if (typeof attempt.lastHeartbeatTime === 'number') {
    return attempt.lastHeartbeatTime;
  }

  if (typeof attempt.lastHeartBeatTime === 'number') {
    return attempt.lastHeartBeatTime;
  }

  if (typeof attempt.startTime === 'number') {
    return attempt.startTime;
  }

  return null;
};

const normalizeAttempt = (value: unknown): Attempt | null => {
  if (value === null || typeof value === 'undefined') {
    return null;
  }

  const camelized = camelCaseKeys(value) as Attempt & {
    lastHeartbeatTime?: Timestamp | null;
    lastHeartBeatTime?: Timestamp | null;
  };
  const { lastHeartbeatTime, lastHeartBeatTime, ...rest } = camelized;

  return {
    ...rest,
    lastHeartbeatTime: normalizeHeartbeat({ ...rest, lastHeartbeatTime, lastHeartBeatTime }),
  };
};

const normalizeAttemptStrict = (value: unknown): Attempt => {
  const normalized = normalizeAttempt(value);
  if (!normalized) {
    throw new Error('Expected attempt payload');
  }
  return normalized;
};

const normalizeRollout = (value: unknown): Rollout => {
  const camelized = camelCaseKeys(value) as Rollout & { attempt?: unknown };
  const { attempt, ...rest } = camelized;

  return {
    ...rest,
    attempt: normalizeAttempt(attempt),
  };
};

const normalizeSpan = (value: unknown): Span => {
  const camelized = camelCaseKeys(value) as Span & {
    status?: {
      status_code?: Span['status']['status_code'];
      statusCode?: Span['status']['status_code'];
      description?: string | null;
    };
  };
  const rawStatus = camelized.status ?? { status_code: 'UNSET', description: null };
  const result = {
    ...camelized,
    parentId: camelized.parentId ?? null,
    // The following fields does not need to be normalized to camel case
    // For example, gen_ai.xxx should not become genAi.xxx
    attributes: (value as any).attributes ?? {},
    context: (value as any).context ?? {},
    parent: (value as any).parent ?? null,
    resource: (value as any).resource ?? {},
    status: {
      status_code: rawStatus.status_code ?? rawStatus.statusCode ?? 'UNSET',
      description: rawStatus.description ?? null,
    },
  };
  return result;
};

const normalizeResources = (value: unknown): Resources => {
  const camelized = camelCaseKeys(value) as Resources;
  return {
    resourcesId: camelized.resourcesId,
    version: camelized.version,
    createTime: camelized.createTime,
    updateTime: camelized.updateTime,
    resources: camelized.resources ?? {},
  };
};

const normalizeWorker = (value: unknown): Worker => {
  const camelized = camelCaseKeys(value) as Worker;
  return {
    workerId: camelized.workerId,
    status: camelized.status,
    heartbeatStats: camelized.heartbeatStats ?? null,
    lastHeartbeatTime: camelized.lastHeartbeatTime ?? null,
    lastDequeueTime: camelized.lastDequeueTime ?? null,
    lastBusyTime: camelized.lastBusyTime ?? null,
    lastIdleTime: camelized.lastIdleTime ?? null,
    currentRolloutId: camelized.currentRolloutId ?? null,
    currentAttemptId: camelized.currentAttemptId ?? null,
  };
};

const normalizePaginatedResponse = <T>(value: unknown, normalizer: (item: unknown) => T): PaginatedResponse<T> => {
  if (!value || typeof value !== 'object') {
    throw new Error('Expected paginated response payload');
  }

  const converted = value as {
    items?: unknown;
    limit?: number;
    offset?: number;
    total?: number;
  };

  const itemsSource = Array.isArray(converted.items) ? converted.items : [];

  return {
    items: itemsSource.map((item) => normalizer(item)),
    limit: typeof converted.limit === 'number' ? converted.limit : itemsSource.length,
    offset: typeof converted.offset === 'number' ? converted.offset : 0,
    total: typeof converted.total === 'number' ? converted.total : itemsSource.length,
  };
};

const dynamicBaseQuery: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = async (
  args,
  api,
  extraOptions,
) => {
  const state = api.getState() as RootState;
  const stateBaseUrl = state.config?.baseUrl;
  const fallbackBaseUrl = typeof window !== 'undefined' ? window.location.origin : '';
  const baseUrl = stateBaseUrl && stateBaseUrl.trim().length > 0 ? stateBaseUrl : fallbackBaseUrl;
  const preparedArgs: FetchArgs =
    typeof args === 'string'
      ? { url: args }
      : {
          ...args,
          url: args.url ?? '',
        };

  const absoluteUrl = buildAbsoluteUrl(baseUrl, preparedArgs.url ?? '');
  return rawBaseQuery({ ...preparedArgs, url: absoluteUrl }, api, extraOptions);
};

export type GetRolloutsQueryArgs = {
  limit: number;
  offset: number;
  sortBy?: string | null;
  sortOrder?: 'asc' | 'desc';
  statusIn?: RolloutStatus[];
  rolloutIdContains?: string | null;
  modeIn?: RolloutMode[];
};

export type GetResourcesQueryArgs = {
  limit: number;
  offset: number;
  sortBy?: string | null;
  sortOrder?: 'asc' | 'desc';
  resourcesIdContains?: string | null;
};

export type GetWorkersQueryArgs = {
  limit: number;
  offset: number;
  sortBy?: string | null;
  sortOrder?: 'asc' | 'desc';
  workerIdContains?: string | null;
  statusIn?: WorkerStatus[];
};

export type GetRolloutAttemptsQueryArgs = {
  rolloutId: string;
  limit?: number;
  offset?: number;
  sortBy?: string | null;
  sortOrder?: 'asc' | 'desc';
};

export type GetSpansQueryArgs = {
  rolloutId: string;
  attemptId?: string | null;
  limit?: number;
  offset?: number;
  sortBy?: string | null;
  sortOrder?: 'asc' | 'desc';
  traceIdContains?: string | null;
  spanIdContains?: string | null;
  parentIdContains?: string | null;
  nameContains?: string | null;
  filterLogic?: 'and' | 'or' | null;
};

export const rolloutsApi = createApi({
  reducerPath: 'rolloutsApi',
  baseQuery: dynamicBaseQuery,
  tagTypes: ['Rollout', 'Span', 'Resources', 'Worker'],
  endpoints: (builder) => ({
    getResources: builder.query<PaginatedResponse<Resources>, GetResourcesQueryArgs>({
      query: ({ limit, offset, sortBy, sortOrder, resourcesIdContains }) => {
        const searchParams = new URLSearchParams();
        searchParams.set('limit', String(typeof limit === 'number' ? limit : -1));
        searchParams.set('offset', String(typeof offset === 'number' ? offset : 0));
        if (sortBy) {
          searchParams.set('sort_by', sortBy);
        }
        if (sortOrder) {
          searchParams.set('sort_order', sortOrder);
        }
        if (resourcesIdContains && resourcesIdContains.trim().length > 0) {
          searchParams.set('resources_id_contains', resourcesIdContains.trim());
        }

        const queryString = searchParams.toString();
        const url = queryString.length > 0 ? `v1/agl/resources?${queryString}` : 'v1/agl/resources';
        return { url, method: 'GET' };
      },
      transformResponse: (response: unknown) => normalizePaginatedResponse(response, normalizeResources),
      providesTags: (result) =>
        result
          ? [
              { type: 'Resources' as const, id: 'LIST' },
              ...result.items.map((item) => ({ type: 'Resources' as const, id: item.resourcesId })),
            ]
          : [{ type: 'Resources' as const, id: 'LIST' }],
    }),
    getWorkers: builder.query<PaginatedResponse<Worker>, GetWorkersQueryArgs>({
      query: ({ limit, offset, sortBy, sortOrder, workerIdContains, statusIn }) => {
        const searchParams = new URLSearchParams();
        searchParams.set('limit', String(typeof limit === 'number' ? limit : -1));
        searchParams.set('offset', String(typeof offset === 'number' ? offset : 0));
        if (sortBy) {
          searchParams.set('sort_by', sortBy);
        }
        if (sortOrder) {
          searchParams.set('sort_order', sortOrder);
        }
        if (workerIdContains && workerIdContains.trim().length > 0) {
          searchParams.set('worker_id_contains', workerIdContains.trim());
        }
        if (statusIn && statusIn.length > 0) {
          statusIn.forEach((status) => searchParams.append('status_in', status));
        }

        const queryString = searchParams.toString();
        const url = queryString.length > 0 ? `v1/agl/workers?${queryString}` : 'v1/agl/workers';
        return { url, method: 'GET' };
      },
      transformResponse: (response: unknown) => normalizePaginatedResponse(response, normalizeWorker),
      providesTags: (result) =>
        result
          ? [
              { type: 'Worker' as const, id: 'LIST' },
              ...result.items.map((worker) => ({ type: 'Worker' as const, id: worker.workerId })),
            ]
          : [{ type: 'Worker' as const, id: 'LIST' }],
    }),
    getRollouts: builder.query<PaginatedResponse<Rollout>, GetRolloutsQueryArgs>({
      query: ({ limit, offset, sortBy, sortOrder, statusIn, rolloutIdContains, modeIn }) => {
        const searchParams = new URLSearchParams();
        searchParams.set('limit', String(typeof limit === 'number' ? limit : -1));
        searchParams.set('offset', String(typeof offset === 'number' ? offset : 0));
        if (sortBy) {
          searchParams.set('sort_by', sortBy);
        }
        if (sortOrder) {
          searchParams.set('sort_order', sortOrder);
        }
        if (statusIn && statusIn.length > 0) {
          statusIn.forEach((status) => searchParams.append('status_in', status));
        }
        if (modeIn && modeIn.length > 0) {
          modeIn.forEach((mode) => searchParams.append('mode_in', mode));
        }
        if (rolloutIdContains && rolloutIdContains.trim().length > 0) {
          searchParams.set('rollout_id_contains', rolloutIdContains.trim());
        }

        const queryString = searchParams.toString();
        const url = queryString.length > 0 ? `v1/agl/rollouts?${queryString}` : 'v1/agl/rollouts';
        return { url, method: 'GET' };
      },
      transformResponse: (response: unknown) => normalizePaginatedResponse(response, normalizeRollout),
      providesTags: (result) =>
        result
          ? [
              { type: 'Rollout' as const, id: 'LIST' },
              ...result.items.map((rollout) => ({ type: 'Rollout' as const, id: rollout.rolloutId })),
            ]
          : [{ type: 'Rollout' as const, id: 'LIST' }],
    }),
    getRolloutAttempts: builder.query<PaginatedResponse<Attempt>, GetRolloutAttemptsQueryArgs>({
      query: ({ rolloutId, limit = -1, offset = 0, sortBy, sortOrder }) => {
        const searchParams = new URLSearchParams();
        searchParams.set('limit', String(typeof limit === 'number' ? limit : -1));
        searchParams.set('offset', String(typeof offset === 'number' ? offset : 0));
        if (sortBy) {
          searchParams.set('sort_by', sortBy);
        }
        if (sortOrder) {
          searchParams.set('sort_order', sortOrder);
        }
        const queryString = searchParams.toString();
        const url =
          queryString.length > 0
            ? `v1/agl/rollouts/${rolloutId}/attempts?${queryString}`
            : `v1/agl/rollouts/${rolloutId}/attempts`;
        return { url, method: 'GET' };
      },
      transformResponse: (response: unknown) => normalizePaginatedResponse(response, normalizeAttemptStrict),
      providesTags: (_result, _error, queryArgs) => [{ type: 'Rollout', id: queryArgs.rolloutId }],
    }),
    getSpans: builder.query<PaginatedResponse<Span>, GetSpansQueryArgs>({
      query: (args) => {
        if (!args.rolloutId) {
          throw new Error('rolloutId is required to fetch spans');
        }
        const searchParams = new URLSearchParams({ rollout_id: args.rolloutId });
        if (args.attemptId) {
          searchParams.set('attempt_id', args.attemptId);
        }
        if (typeof args.limit === 'number') {
          searchParams.set('limit', String(args.limit));
        }
        if (typeof args.offset === 'number') {
          searchParams.set('offset', String(args.offset));
        }
        if (args.sortBy) {
          searchParams.set('sort_by', args.sortBy);
        }
        if (args.sortOrder) {
          searchParams.set('sort_order', args.sortOrder);
        }
        if (args.traceIdContains) {
          searchParams.set('trace_id_contains', args.traceIdContains);
        }
        if (args.spanIdContains) {
          searchParams.set('span_id_contains', args.spanIdContains);
        }
        if (args.parentIdContains) {
          searchParams.set('parent_id_contains', args.parentIdContains);
        }
        if (args.nameContains) {
          searchParams.set('name_contains', args.nameContains);
        }
        if (args.filterLogic) {
          searchParams.set('filter_logic', args.filterLogic);
        }
        return { url: `v1/agl/spans?${searchParams.toString()}`, method: 'GET' };
      },
      transformResponse: (response: unknown) => normalizePaginatedResponse(response, normalizeSpan),
      providesTags: (_result, _error, args) =>
        args
          ? [
              { type: 'Span' as const, id: `${args.rolloutId}:${args.attemptId ?? 'latest'}` },
              { type: 'Span' as const, id: 'LIST' },
            ]
          : [{ type: 'Span' as const, id: 'LIST' }],
    }),
  }),
});

export const {
  useGetResourcesQuery,
  useGetWorkersQuery,
  useGetRolloutsQuery,
  useGetRolloutAttemptsQuery,
  useGetSpansQuery,
} = rolloutsApi;
