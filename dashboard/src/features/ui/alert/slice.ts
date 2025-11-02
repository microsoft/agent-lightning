// Copyright (c) Microsoft. All rights reserved.

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type AlertTone = 'info' | 'warning' | 'error';

export type AppAlert = {
  id: string;
  message: string;
  tone: AlertTone;
  isVisible: boolean;
  createdAt: number;
};

export type AlertsState = {
  alerts: AppAlert[];
};

const initialState: AlertsState = {
  alerts: [],
};

type ShowAlertPayload = {
  id: string;
  message: string;
  tone?: AlertTone;
};

type HideAlertPayload = {
  id: string;
};

const alertsSlice = createSlice({
  name: 'alert',
  initialState,
  reducers: {
    showAlert(state, action: PayloadAction<ShowAlertPayload>) {
      const { id, message, tone = 'info' } = action.payload;
      const existing = state.alerts.find((alert) => alert.id === id);

      if (existing) {
        existing.message = message;
        existing.tone = tone;
        existing.isVisible = true;
        existing.createdAt = Date.now();
      } else {
        state.alerts.push({
          id,
          message,
          tone,
          isVisible: true,
          createdAt: Date.now(),
        });
      }
    },
    hideAlert(state, action: PayloadAction<HideAlertPayload>) {
      state.alerts = state.alerts.filter((alert) => alert.id !== action.payload.id);
    },
    dismissAlert(state, action: PayloadAction<HideAlertPayload>) {
      const entry = state.alerts.find((alert) => alert.id === action.payload.id);
      if (entry) {
        entry.isVisible = false;
      }
    },
    clearAlerts(state) {
      state.alerts = [];
    },
  },
});

export const { showAlert, hideAlert, dismissAlert, clearAlerts } = alertsSlice.actions;
export const alertReducer = alertsSlice.reducer;
