// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from '@/features/rollouts';
import type { Resources } from '@/types';
import { selectResourcesQueryArgs } from './selectors';
import {
  resetResourcesFilters,
  setResourcesPage,
  setResourcesRecordsPerPage,
  setResourcesSearchTerm,
  setResourcesSort,
} from './slice';

const extractResourceIds = (resources: Resources[]): string[] => resources.map((resource) => resource.resourcesId);

describe('resources feature integration', () => {
  it('builds default query arguments from the UI state', () => {
    const store = createServerBackedStore();
    const queryArgs = selectResourcesQueryArgs(store.getState());

    expect(queryArgs).toMatchObject({
      limit: 50,
      offset: 0,
      sortBy: 'update_time',
      sortOrder: 'desc',
      resourcesIdContains: undefined,
    });
  });

  it('fetches resources from the Python LightningStore server', async () => {
    const store = createServerBackedStore();
    const queryArgs = selectResourcesQueryArgs(store.getState());

    const subscription = store.dispatch(rolloutsApi.endpoints.getResources.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.total).toBe(5);
    expect(data.items).toHaveLength(5);

    const resourceIds = extractResourceIds(data.items);
    expect(resourceIds).toEqual(expect.arrayContaining(['rs-story-001', 'rs-story-005']));

    const updateTimes = data.items.map((resource) => resource.updateTime);
    const sortedUpdateTimes = [...updateTimes].sort((a, b) => b - a);
    expect(updateTimes).toEqual(sortedUpdateTimes);

    expect(data.items[0].resources).toBeDefined();
    expect(Object.keys(data.items[0].resources)).not.toHaveLength(0);
  });

  it('paginates resource results based on UI state', async () => {
    const store = createServerBackedStore();
    store.dispatch(setResourcesRecordsPerPage(2));
    store.dispatch(setResourcesPage(2));

    const queryArgs = selectResourcesQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({ limit: 2, offset: 2 });

    const subscription = store.dispatch(rolloutsApi.endpoints.getResources.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(2);
    expect(data.total).toBe(5);
    expect(data.items.map((resource) => resource.resourcesId)).toEqual(['rs-story-005', 'rs-story-002']);
  });

  it('applies search and sorting preferences', async () => {
    const store = createServerBackedStore();
    store.dispatch(resetResourcesFilters());
    store.dispatch(setResourcesSearchTerm('rs-story-003'));
    store.dispatch(setResourcesSort({ column: 'version', direction: 'asc' }));

    const queryArgs = selectResourcesQueryArgs(store.getState());
    expect(queryArgs).toMatchObject({
      limit: 50,
      offset: 0,
      sortBy: 'version',
      sortOrder: 'asc',
      resourcesIdContains: 'rs-story-003',
    });

    const subscription = store.dispatch(rolloutsApi.endpoints.getResources.initiate(queryArgs));
    const data = await subscription.unwrap();
    subscription.unsubscribe();

    expect(data.items).toHaveLength(1);
    expect(data.items[0].resourcesId).toBe('rs-story-003');
    expect(data.items[0].version).toBeGreaterThan(0);
  });
});
