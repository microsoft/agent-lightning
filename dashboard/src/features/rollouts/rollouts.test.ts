// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from './api';
import { selectRolloutsQueryArgs } from './selectors';

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
});
