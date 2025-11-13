// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from '@/features/rollouts';
import type { Worker } from '@/types';
import { selectWorkersQueryArgs } from './selectors';
import {
  resetWorkersFilters,
  setWorkersPage,
  setWorkersRecordsPerPage,
  setWorkersSearchTerm,
  setWorkersSort,
} from './slice';

const extractWorkerIds = (workers: Worker[]): string[] => workers.map((worker) => worker.workerId);

describe('workers feature integration', () => {
  it('builds default query arguments from the UI state', () => {
    const store = createServerBackedStore();
    const queryArgs = selectWorkersQueryArgs(store.getState());

    expect(queryArgs).toMatchObject({
      limit: 50,
      offset: 0,
      sortBy: 'last_heartbeat_time',
      sortOrder: 'desc',
      workerIdContains: undefined,
    });
  });

  it('fetches workers from the Python LightningStore server', async () => {
    const store = createServerBackedStore();
    const queryArgs = selectWorkersQueryArgs(store.getState());

    const subscription = store.dispatch(rolloutsApi.endpoints.getWorkers.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.total).toBeGreaterThanOrEqual(4);
    expect(data.items).toHaveLength(Math.min(queryArgs.limit, data.total));

    const workerIds = extractWorkerIds(data.items);
    expect(workerIds).toEqual(expect.arrayContaining(['worker-east', 'worker-west']));

    const heartbeatTimes = data.items.map((worker) => worker.lastHeartbeatTime ?? 0);
    const sortedHeartbeatTimes = [...heartbeatTimes].sort((a, b) => b - a);
    expect(heartbeatTimes).toEqual(sortedHeartbeatTimes);
  });

  it('paginates worker results based on UI state', async () => {
    const store = createServerBackedStore();
    store.dispatch(setWorkersRecordsPerPage(2));
    store.dispatch(setWorkersPage(2));

    const queryArgs = selectWorkersQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({ limit: 2, offset: 2 });

    const subscription = store.dispatch(rolloutsApi.endpoints.getWorkers.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(2);
    expect(data.total).toBeGreaterThanOrEqual(4);
  });

  it('applies search and sorting preferences', async () => {
    const store = createServerBackedStore();
    store.dispatch(resetWorkersFilters());
    store.dispatch(setWorkersSearchTerm('worker-west'));
    store.dispatch(setWorkersSort({ column: 'workerId', direction: 'asc' }));

    const queryArgs = selectWorkersQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({
      limit: 50,
      offset: 0,
      sortBy: 'worker_id',
      sortOrder: 'asc',
      workerIdContains: 'worker-west',
    });

    const subscription = store.dispatch(rolloutsApi.endpoints.getWorkers.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(1);
    expect(data.items[0].workerId).toBe('worker-west');
  });

  it('maps current rollout/attempt sorting to backend fields', () => {
    const store = createServerBackedStore();
    store.dispatch(setWorkersSort({ column: 'currentRolloutId', direction: 'desc' }));
    let queryArgs = selectWorkersQueryArgs(store.getState());
    expect(queryArgs.sortBy).toBe('current_rollout_id');
    expect(queryArgs.sortOrder).toBe('desc');

    store.dispatch(setWorkersSort({ column: 'currentAttemptId', direction: 'asc' }));
    queryArgs = selectWorkersQueryArgs(store.getState());
    expect(queryArgs.sortBy).toBe('current_attempt_id');
    expect(queryArgs.sortOrder).toBe('asc');
  });
});
