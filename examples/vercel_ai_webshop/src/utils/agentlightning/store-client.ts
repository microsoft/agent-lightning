/**
 * Agent Lightning Store REST Client
 *
 * HTTP client for communicating with the Agent Lightning Store server.
 * Based on endpoints defined in agentlightning/store/client_server.py.
 */

import type {
  AttemptedRollout,
  Attempt,
  Rollout,
  StartRolloutRequest,
  UpdateAttemptRequest,
  AttemptStatus,
  TaskInput,
  PaginatedResult,
  ResourcesUpdate,
} from './types';

const API_V1_AGL_PREFIX = '/v1/agl';

export interface StoreClientOptions {
  baseUrl: string;
  timeoutMs?: number;
}

export class AgentLightningStoreClient {
  private baseUrl: string;
  private timeoutMs: number;

  constructor(options: StoreClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  get isConfigured(): boolean {
    return Boolean(this.baseUrl);
  }

  private get apiBase(): string {
    return `${this.baseUrl}${API_V1_AGL_PREFIX}`;
  }

  /**
   * Check if the Store server is healthy.
   */
  async health(): Promise<boolean> {
    try {
      const response = await this.get('/health');
      return (response as { status: string }).status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * Start a new rollout immediately (POST /v1/agl/rollouts).
   * Returns an AttemptedRollout with the initial attempt already started.
   */
  async startRollout(request: StartRolloutRequest): Promise<AttemptedRollout> {
    const response = await this.post('/rollouts', request);
    return response as AttemptedRollout;
  }

  /**
   * Enqueue rollouts for later processing (POST /v1/agl/queues/rollouts/enqueue).
   */
  async enqueueRollouts(
    rollouts: Array<{
      input: TaskInput;
      mode?: 'train' | 'val' | 'test';
      metadata?: Record<string, unknown>;
    }>
  ): Promise<Rollout[]> {
    const response = await this.post('/queues/rollouts/enqueue', { rollouts });
    return response as Rollout[];
  }

  /**
   * Dequeue available rollouts (POST /v1/agl/queues/rollouts/dequeue).
   */
  async dequeueRollouts(
    limit: number = 1,
    workerId?: string
  ): Promise<AttemptedRollout[]> {
    const payload: Record<string, unknown> = { limit };
    if (workerId) payload.worker_id = workerId;
    const response = await this.post('/queues/rollouts/dequeue', payload);
    return response as AttemptedRollout[];
  }

  /**
   * Get a rollout by ID (GET /v1/agl/rollouts/{rollout_id}).
   */
  async getRollout(rolloutId: string): Promise<Rollout | AttemptedRollout | null> {
    try {
      const response = await this.get(`/rollouts/${encodeURIComponent(rolloutId)}`);
      return response as Rollout | AttemptedRollout;
    } catch (error) {
      if (error instanceof Error && error.message.includes('404')) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Update rollout metadata (POST /v1/agl/rollouts/{rollout_id}).
   */
  async updateRollout(
    rolloutId: string,
    update: {
      status?: string;
      metadata?: Record<string, unknown>;
    }
  ): Promise<Rollout> {
    const response = await this.post(
      `/rollouts/${encodeURIComponent(rolloutId)}`,
      update
    );
    return response as Rollout;
  }

  /**
   * Start a new attempt for a rollout (POST /v1/agl/rollouts/{rollout_id}/attempts).
   */
  async startAttempt(
    rolloutId: string,
    workerId?: string
  ): Promise<AttemptedRollout> {
    const payload = workerId ? { worker_id: workerId } : undefined;
    const response = await this.post(
      `/rollouts/${encodeURIComponent(rolloutId)}/attempts`,
      payload
    );
    return response as AttemptedRollout;
  }

  /**
   * Update attempt status/metadata (POST /v1/agl/rollouts/{rollout_id}/attempts/{attempt_id}).
   */
  async updateAttempt(
    rolloutId: string,
    attemptId: string,
    update: UpdateAttemptRequest
  ): Promise<Attempt> {
    const response = await this.post(
      `/rollouts/${encodeURIComponent(rolloutId)}/attempts/${encodeURIComponent(attemptId)}`,
      update
    );
    return response as Attempt;
  }

  /**
   * Get all attempts for a rollout (GET /v1/agl/rollouts/{rollout_id}/attempts).
   */
  async getAttempts(rolloutId: string): Promise<PaginatedResult<Attempt>> {
    const response = await this.get(
      `/rollouts/${encodeURIComponent(rolloutId)}/attempts`
    );
    return response as PaginatedResult<Attempt>;
  }

  /**
   * Get the latest attempt for a rollout (GET /v1/agl/rollouts/{rollout_id}/attempts/latest).
   */
  async getLatestAttempt(rolloutId: string): Promise<Attempt | null> {
    try {
      const response = await this.get(
        `/rollouts/${encodeURIComponent(rolloutId)}/attempts/latest`
      );
      return response as Attempt | null;
    } catch {
      return null;
    }
  }

  /**
   * Convenience method: Complete an attempt with success/failure status.
   */
  async completeAttempt(
    rolloutId: string,
    attemptId: string,
    options: {
      success: boolean;
      reward?: number;
      error?: string;
    }
  ): Promise<Attempt> {
    const status: AttemptStatus = options.success ? 'succeeded' : 'failed';
    return this.updateAttempt(rolloutId, attemptId, {
      status,
      metadata: {
        reward: options.reward,
        success: options.success,
        error_message: options.error,
        completed_at: Date.now(),
      },
    });
  }

  /**
   * Update worker heartbeat (POST /v1/agl/workers/{worker_id}).
   */
  async updateWorker(
    workerId: string,
    heartbeatStats?: Record<string, unknown>
  ): Promise<unknown> {
    return this.post(`/workers/${encodeURIComponent(workerId)}`, {
      heartbeat_stats: heartbeatStats,
    });
  }

  // =========================================================================
  // Resource Methods
  // =========================================================================

  /**
   * Get the latest resources (GET /v1/agl/resources/latest).
   * Returns the most recently published resource configuration.
   */
  async getLatestResources(): Promise<ResourcesUpdate | null> {
    try {
      const response = await this.get('/resources/latest');
      return response as ResourcesUpdate | null;
    } catch {
      return null;
    }
  }

  /**
   * Get resources by ID (GET /v1/agl/resources/{resources_id}).
   */
  async getResourcesById(resourcesId: string): Promise<ResourcesUpdate | null> {
    try {
      const response = await this.get(
        `/resources/${encodeURIComponent(resourcesId)}`
      );
      return response as ResourcesUpdate | null;
    } catch (error) {
      if (error instanceof Error && error.message.includes('404')) {
        return null;
      }
      throw error;
    }
  }

  // HTTP helpers

  private async post(path: string, body?: unknown): Promise<unknown> {
    const url = `${this.apiBase}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`Store error ${res.status}: ${text.slice(0, 500)}`);
      }

      const contentType = res.headers.get('content-type') ?? '';
      if (contentType.includes('application/json')) {
        return await res.json();
      }
      return await res.text();
    } finally {
      clearTimeout(timer);
    }
  }

  private async get(path: string): Promise<unknown> {
    const url = `${this.apiBase}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(url, {
        method: 'GET',
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`Store error ${res.status}: ${text.slice(0, 500)}`);
      }

      const contentType = res.headers.get('content-type') ?? '';
      if (contentType.includes('application/json')) {
        return await res.json();
      }
      return await res.text();
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * Get a Store client instance from environment variables.
 * Returns null if AGENT_LIGHTNING_STORE_URL is not configured.
 */
export function getStoreClient(): AgentLightningStoreClient | null {
  const baseUrl = process.env.AGENT_LIGHTNING_STORE_URL;
  if (!baseUrl) return null;
  return new AgentLightningStoreClient({ baseUrl });
}
