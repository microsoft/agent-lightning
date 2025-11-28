// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from './api';
import { selectRolloutsQueryArgs } from './selectors';
import {
  resetRolloutsFilters,
  setRolloutsModeFilters,
  setRolloutsPage,
  setRolloutsRecordsPerPage,
  setRolloutsSearchTerm,
  setRolloutsSort,
  setRolloutsStatusFilters,
} from './slice';

describe('rollouts feature integration', () => {
  it('builds default query arguments from the UI state', () => {
    const store = createServerBackedStore();
    const queryArgs = selectRolloutsQueryArgs(store.getState());

    expect(queryArgs).toMatchObject({
      limit: 100,
      offset: 0,
      sortBy: 'start_time',
      sortOrder: 'desc',
      rolloutIdContains: undefined,
      statusIn: undefined,
      modeIn: undefined,
    });
  });

  it('retrieves rollouts from the Python LightningStore server', async () => {
    const store = createServerBackedStore();
    const queryArgs = selectRolloutsQueryArgs(store.getState());

    const subscription = store.dispatch(rolloutsApi.endpoints.getRollouts.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.total).toBe(6);
    expect(data.items).toHaveLength(6);

    const rolloutIds = data.items.map((rollout) => rollout.rolloutId);
    expect(rolloutIds).toEqual(
      expect.arrayContaining(['ro-story-001', 'ro-story-002', 'ro-story-003', 'ro-story-004', 'ro-story-005']),
    );

    const startTimes = data.items.map((rollout) => rollout.startTime);
    const sortedStartTimes = [...startTimes].sort((a, b) => b - a);
    expect(startTimes).toEqual(sortedStartTimes);
    expect(data.items[0].rolloutId).toBe('ro-story-005');
    expect(data.items[0].status).toBeDefined();
  });

  it('includes attempts directly on rollout payloads when they exist', async () => {
    const store = createServerBackedStore();
    const queryArgs = selectRolloutsQueryArgs(store.getState());

    const subscription = store.dispatch(rolloutsApi.endpoints.getRollouts.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    const rolloutWithAttempt = data.items.find((rollout) => rollout.rolloutId === 'ro-story-002');
    expect(rolloutWithAttempt).toBeDefined();
    expect(rolloutWithAttempt?.attempt).not.toBeNull();
    expect(rolloutWithAttempt?.attempt?.attemptId).toBe('at-story-022');

    const rolloutWithoutAttempt = data.items.find((rollout) => rollout.rolloutId === 'ro-story-004');
    expect(rolloutWithoutAttempt).toBeDefined();
    expect(rolloutWithoutAttempt?.attempt).toBeNull();
  });

  it('retrieves attempts for a rollout from the Python server', async () => {
    const store = createServerBackedStore();
    const subscription = store.dispatch(
      rolloutsApi.endpoints.getRolloutAttempts.initiate({ rolloutId: 'ro-story-002' }),
    );
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.total).toBe(2);
    expect(data.items.map((attempt) => attempt.attemptId)).toEqual(['at-story-021', 'at-story-022']);
  });

  it('paginates rollouts with custom UI state', async () => {
    const store = createServerBackedStore();
    store.dispatch(setRolloutsRecordsPerPage(2));
    store.dispatch(setRolloutsPage(2));

    const queryArgs = selectRolloutsQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({ limit: 2, offset: 2 });

    const subscription = store.dispatch(rolloutsApi.endpoints.getRollouts.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(2);
    expect(data.items.map((rollout) => rollout.rolloutId)).toEqual(['ro-story-004', 'ro-story-002']);
  });

  it('filters and sorts rollouts based on UI selections', async () => {
    const store = createServerBackedStore();
    store.dispatch(resetRolloutsFilters());
    store.dispatch(setRolloutsStatusFilters(['succeeded']));
    store.dispatch(setRolloutsModeFilters(['val']));
    store.dispatch(setRolloutsSearchTerm('ro-story-002'));
    store.dispatch(setRolloutsSort({ column: 'rolloutId', direction: 'asc' }));

    const queryArgs = selectRolloutsQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({
      statusIn: ['succeeded'],
      modeIn: ['val'],
      rolloutIdContains: 'ro-story-002',
      sortBy: 'rollout_id',
      sortOrder: 'asc',
    });

    const subscription = store.dispatch(rolloutsApi.endpoints.getRollouts.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(1);
    expect(data.items[0].rolloutId).toBe('ro-story-002');
    expect(data.items[0].status).toBe('succeeded');
  });
});
