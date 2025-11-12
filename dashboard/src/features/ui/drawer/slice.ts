// Copyright (c) Microsoft. All rights reserved.

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { Attempt, Rollout, Span, Worker } from '@/types';

export type DrawerType = 'rollout-json' | 'rollout-traces' | 'trace-detail' | 'worker-detail';

export type DrawerContent =
  | {
      type: 'rollout-json' | 'rollout-traces';
      rollout: Rollout;
      attempt: Attempt | null;
      isNested: boolean;
    }
  | {
      type: 'trace-detail';
      span: Span;
      rollout: Rollout | null;
      attempt: Attempt | null;
    }
  | {
      type: 'worker-detail';
      worker: Worker;
    };

export type DrawerState = {
  isOpen: boolean;
  content: DrawerContent | null;
};

export const initialDrawerState: DrawerState = {
  isOpen: false,
  content: null,
};

const drawerSlice = createSlice({
  name: 'drawer',
  initialState: initialDrawerState,
  reducers: {
    openDrawer(state, action: PayloadAction<DrawerContent>) {
      state.isOpen = true;
      state.content = action.payload;
    },
    closeDrawer(state) {
      state.isOpen = false;
      state.content = null;
    },
  },
});

export const { openDrawer, closeDrawer } = drawerSlice.actions;

export const drawerReducer = drawerSlice.reducer;
