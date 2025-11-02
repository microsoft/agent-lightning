// Copyright (c) Microsoft. All rights reserved.

import type { RootState } from '@/store';

export const selectDrawerState = (state: RootState) => state.drawer;

export const selectDrawerIsOpen = (state: RootState) => selectDrawerState(state).isOpen;

export const selectDrawerContent = (state: RootState) => selectDrawerState(state).content;
