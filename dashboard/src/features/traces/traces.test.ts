// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from '../rollouts';
import { selectTracesQueryArgs } from './selectors';
import { setTracesPage, setTracesRecordsPerPage, setTracesRolloutId, setTracesSearchTerm } from './slice';

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

  it('paginates spans based on UI state', async () => {
    const store = createServerBackedStore();
    store.dispatch(setTracesRolloutId('ro-story-001'));
    store.dispatch(setTracesRecordsPerPage(2));

    const firstPageArgs = selectTracesQueryArgs(store.getState());
    expect(firstPageArgs).toMatchObject({ limit: 2, offset: 0 });

    const firstPageSub = store.dispatch(rolloutsApi.endpoints.getSpans.initiate(firstPageArgs!));
    const firstPage = await firstPageSub.unwrap();
    firstPageSub.unsubscribe();

    expect(firstPage.items.map((span) => span.spanId)).toEqual(['span-003-tool', 'span-002-llm']);

    store.dispatch(setTracesPage(2));
    const secondPageArgs = selectTracesQueryArgs(store.getState());
    expect(secondPageArgs).toMatchObject({ limit: 2, offset: 2 });

    const secondPageSub = store.dispatch(rolloutsApi.endpoints.getSpans.initiate(secondPageArgs!));
    const secondPage = await secondPageSub.unwrap();
    secondPageSub.unsubscribe();

    expect(secondPage.items.map((span) => span.spanId)).toEqual(['span-001-root']);
    expect(secondPage.total).toBe(3);
  });

  it('filters spans using the search term', async () => {
    const store = createServerBackedStore();
    store.dispatch(setTracesRolloutId('ro-story-001'));
    store.dispatch(setTracesSearchTerm('span-003'));

    const queryArgs = selectTracesQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({
      traceIdContains: 'span-003',
      spanIdContains: 'span-003',
      nameContains: 'span-003',
      filterLogic: 'or',
    });

    const subscription = store.dispatch(rolloutsApi.endpoints.getSpans.initiate(queryArgs!));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(1);
    expect(data.items[0].spanId).toBe('span-003-tool');
  });
});
