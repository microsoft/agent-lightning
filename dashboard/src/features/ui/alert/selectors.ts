// Copyright (c) Microsoft. All rights reserved.

import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from '@/store';
import type { AlertsState, AlertTone } from './slice';

const ALERT_PRIORITY: Record<AlertTone, number> = {
  error: 3,
  warning: 2,
  info: 1,
};

const selectAlertState = (state: RootState): AlertsState => state.alert;

export const selectVisibleAlerts = createSelector(selectAlertState, (state) =>
  state.alerts.filter((alert) => alert.isVisible),
);

export const selectHighestPriorityAlert = createSelector(selectVisibleAlerts, (alerts) => {
  if (alerts.length === 0) {
    return null;
  }

  return alerts
    .slice()
    .sort((a, b) => {
      const priorityDiff = ALERT_PRIORITY[a.tone] - ALERT_PRIORITY[b.tone];
      if (priorityDiff !== 0) {
        return priorityDiff;
      }
      return a.createdAt - b.createdAt;
    })
    .at(-1)!;
});
