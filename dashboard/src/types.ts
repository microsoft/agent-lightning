// Copyright (c) Microsoft. All rights reserved.

// This file should sync with agentlightning/types/core.py

export type RolloutStatus = 'queuing' | 'preparing' | 'running' | 'failed' | 'succeeded' | 'cancelled' | 'requeuing';

export type AttemptStatus = 'preparing' | 'running' | 'failed' | 'succeeded' | 'unresponsive' | 'timeout';

export type RolloutMode = 'train' | 'val' | 'test';

export type TaskInput = any;

export type Timestamp = number;

/**
 * Synced with agentlightning.types.core.Attempt
 * with camel case and snake case conversions
 */
export type Attempt = {
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  startTime: Timestamp;
  endTime: Timestamp | null;
  status: AttemptStatus;
  workerId: string | null;
  lastHeartbeatTime: Timestamp | null;
  metadata: Record<string, any> | null;
};

export type WorkerStatus = 'idle' | 'busy' | 'unknown';

/**
 * Synced with agentlightning.types.core.Worker
 * with camel case and snake case conversions
 */
export type Worker = {
  workerId: string;
  status: WorkerStatus;
  heartbeatStats: Record<string, any> | null;
  lastHeartbeatTime: Timestamp | null;
  lastDequeueTime: Timestamp | null;
  lastBusyTime: Timestamp | null;
  lastIdleTime: Timestamp | null;
  currentRolloutId: string | null;
  currentAttemptId: string | null;
};

/**
 * Synced with agentlightning.types.core.Rollout
 * with camel case and snake case conversions
 *
 * The `attempt` field is from `AttemptedRollout` class,
 * which is the latest attempt of the rollout, if any.
 */
export type Rollout = {
  rolloutId: string;
  input: TaskInput;
  startTime: Timestamp;
  endTime: Timestamp | null;
  mode: RolloutMode | null;
  resourcesId: string | null;
  status: RolloutStatus;
  config: Record<string, any>;
  metadata: Record<string, any> | null;

  attempt: Attempt | null;
};

export type Resource = Record<string, any>;

/**
 * Synced with agentlightning.types.resources.Resources
 * with camel case and snake case conversions
 */
export type Resources = {
  resourcesId: string;
  version: number;
  createTime: Timestamp;
  updateTime: Timestamp;
  resources: Record<string, Resource>;
};

/**
 * Paginated response structure matching the Python server's PaginatedResponse
 */
export type PaginatedResponse<T> = {
  items: T[];
  limit: number;
  offset: number;
  total: number;
};

/**
 * Synced with agentlightning.types.traces.Span
 * with camel case and snake case conversions
 */
export type Span = {
  rolloutId: string;
  attemptId: string;
  sequenceId: number;
  traceId: string;
  spanId: string;
  parentId: string | null;
  name: string;
  status: { status_code: 'UNSET' | 'OK' | 'ERROR'; description: string | null };
  attributes: Record<string, any>;
  startTime: Timestamp;
  endTime: Timestamp;

  // The fields below are less frequently used
  events: any;
  links: any;
  context: any;
  parent: any;
  resource: any;
};

// Configs like server connection
export type Config = {
  baseUrl: string; // e.g. http://localhost:8000
  autoRefreshMs: number; // polling/refresh interval
  theme: 'light' | 'dark';
};

export type ThemePreference = 'light' | 'dark' | 'system';

export type ConfigState = {
  baseUrl: string;
  autoRefreshMs: number;
  theme: ThemePreference;
};
