/**
 * Agent Lightning TypeScript Types
 *
 * Type definitions matching the Python models in agentlightning/types/core.py.
 * Used by the Store client to communicate with the Agent Lightning REST API.
 */

/**
 * Possible rollout modes for training/evaluation.
 */
export type RolloutMode = 'train' | 'val' | 'test';

/**
 * Status of a rollout through its lifecycle.
 */
export type RolloutStatus =
  | 'queuing' // initial status
  | 'preparing' // after the trace is claimed
  | 'running' // after receiving the first trace
  | 'failed' // crashed
  | 'succeeded' // status OK
  | 'cancelled' // cancelled by user (or watchdog)
  | 'requeuing'; // retrying

/**
 * Status of an execution attempt.
 */
export type AttemptStatus =
  | 'preparing'
  | 'running'
  | 'failed'
  | 'succeeded'
  | 'unresponsive' // worker has not reported results for a while
  | 'timeout'; // worker has been working too long

/**
 * Task input type - accepts arbitrary payloads.
 */
export type TaskInput = Record<string, unknown>;

/**
 * Configuration controlling rollout retries and timeouts.
 */
export interface RolloutConfig {
  timeout_seconds?: number | null;
  unresponsive_seconds?: number | null;
  max_attempts?: number;
  retry_condition?: AttemptStatus[];
}

/**
 * Execution attempt for a rollout, including metadata for retries.
 */
export interface Attempt {
  rollout_id: string;
  attempt_id: string;
  sequence_id: number;
  start_time: number;
  end_time?: number | null;
  status: AttemptStatus;
  worker_id?: string | null;
  last_heartbeat_time?: number | null;
  metadata?: Record<string, unknown> | null;
}

/**
 * A rollout represents a unit of work to be executed by an agent.
 */
export interface Rollout {
  rollout_id: string;
  input: TaskInput;
  start_time: number;
  end_time?: number | null;
  mode?: RolloutMode | null;
  resources_id?: string | null;
  status: RolloutStatus;
  config: RolloutConfig;
  metadata?: Record<string, unknown> | null;
}

/**
 * Rollout paired with the currently active attempt.
 */
export interface AttemptedRollout extends Rollout {
  attempt: Attempt;
}

/**
 * Request payload for starting a new rollout.
 */
export interface StartRolloutRequest {
  input: TaskInput;
  mode?: RolloutMode;
  resources_id?: string;
  config?: RolloutConfig;
  metadata?: Record<string, unknown>;
  worker_id?: string;
}

/**
 * Request payload for enqueueing a rollout.
 */
export interface EnqueueRolloutRequest {
  input: TaskInput;
  mode?: RolloutMode;
  resources_id?: string;
  config?: RolloutConfig;
  metadata?: Record<string, unknown>;
}

/**
 * Request payload for updating an attempt.
 */
export interface UpdateAttemptRequest {
  status?: AttemptStatus;
  worker_id?: string;
  last_heartbeat_time?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Request payload for dequeueing rollouts.
 */
export interface DequeueRolloutsRequest {
  limit?: number;
  worker_id?: string;
}

/**
 * Worker status type.
 */
export type WorkerStatus = 'idle' | 'busy' | 'unknown';

/**
 * Worker information.
 */
export interface Worker {
  worker_id: string;
  status: WorkerStatus;
  heartbeat_stats?: Record<string, unknown> | null;
  last_heartbeat_time?: number | null;
  last_dequeue_time?: number | null;
  last_busy_time?: number | null;
  last_idle_time?: number | null;
  current_rollout_id?: string | null;
  current_attempt_id?: string | null;
}

/**
 * Paginated result wrapper.
 */
export interface PaginatedResult<T> {
  items: T[];
  limit: number;
  offset: number;
  total: number;
}

/**
 * WebShop task input format.
 */
export interface WebShopTaskInput {
  task_id: string;
  instruction: string;
  target_attributes?: Record<string, unknown>;
}

// =============================================================================
// Resource Types (matching Python agentlightning/types/resources.py)
// =============================================================================

/**
 * Base LLM resource that identifies an LLM endpoint and its configuration.
 */
export interface LLMResource {
  resource_type: 'llm';
  endpoint: string;
  model: string;
  api_key?: string | null;
  sampling_parameters?: Record<string, unknown>;
}

/**
 * LLM resource that rewrites endpoints through LLMProxy.
 * The proxy injects rollout- and attempt-specific routing information into the
 * endpoint so that downstream services can attribute requests correctly.
 */
export interface ProxyLLMResource {
  resource_type: 'proxy_llm';
  endpoint: string;
  model: string;
  api_key?: string | null;
  sampling_parameters?: Record<string, unknown>;
}

/**
 * Prompt template resource.
 */
export interface PromptTemplateResource {
  resource_type: 'prompt_template';
  template: string;
  engine: 'jinja' | 'f-string' | 'poml';
}

/**
 * Union of all resource types.
 */
export type ResourceUnion = LLMResource | ProxyLLMResource | PromptTemplateResource;

/**
 * Mapping from resource names to their configured instances.
 */
export type NamedResources = Record<string, ResourceUnion>;

/**
 * Update payload broadcast to clients when resources change.
 */
export interface ResourcesUpdate {
  resources_id: string;
  create_time: number;
  update_time: number;
  version: number;
  resources: NamedResources;
}
