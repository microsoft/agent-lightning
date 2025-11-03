// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from '../rollouts';
import { selectTracesQueryArgs } from './selectors';
import { setTracesRolloutId, setTracesSearchTerm } from './slice';

describe('traces feature integration', () => {
  it('requires a rollout id before building query arguments', () => {
    const store = createServerBackedStore();
    expect(selectTracesQueryArgs(store.getState())).toBeUndefined();
  });

  it('builds query arguments when targeting a rollout', () => {
    const store = createServerBackedStore();
    store.dispatch(setTracesRolloutId('ro-story-001'));

    const queryArgs = selectTracesQueryArgs(store.getState());
    expect(queryArgs).toBeDefined();
    expect(queryArgs).toMatchObject({
      rolloutId: 'ro-story-001',
      limit: 100,
      offset: 0,
      sortBy: 'start_time',
      sortOrder: 'desc',
      filterLogic: undefined,
    });

    store.dispatch(setTracesSearchTerm('span-00'));
    const filteredArgs = selectTracesQueryArgs(store.getState());
    expect(filteredArgs?.filterLogic).toBe('or');
  });

  it('fetches spans from the Python LightningStore server', async () => {
    const store = createServerBackedStore();
    store.dispatch(setTracesRolloutId('ro-story-001'));

    const queryArgs = selectTracesQueryArgs(store.getState());
    expect(queryArgs).toBeDefined();

    const subscription = store.dispatch(rolloutsApi.endpoints.getSpans.initiate(queryArgs!));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.total).toBe(3);
    const spanIds = data.items.map((span) => span.spanId).sort();
    expect(spanIds).toEqual(['span-001-root', 'span-002-llm', 'span-003-tool']);
    expect(new Set(data.items.map((span) => span.status.status_code))).toEqual(new Set(['OK']));
  });
});
