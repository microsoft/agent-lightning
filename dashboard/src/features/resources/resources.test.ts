// Copyright (c) Microsoft. All rights reserved.

import { createServerBackedStore } from '@test-utils';
import { describe, expect, it } from 'vitest';
import { rolloutsApi } from '@/features/rollouts';
import type { Resources } from '@/types';
import { selectResourcesQueryArgs } from './selectors';

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
});
