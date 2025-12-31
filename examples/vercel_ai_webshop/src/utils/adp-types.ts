/**
 * Agent Data Protocol (ADP) Types
 *
 * Standardized format for agent trajectories, based on the Agent Data Protocol
 * specification (https://arxiv.org/abs/2510.24702).
 *
 * ADP provides a unified representation for agent interactions that can be
 * converted to various fine-tuning formats (OpenAI, Anthropic, etc.).
 */

/**
 * API Action - represents a tool/function call.
 */
export interface ADPApiAction {
  type: 'api_action';
  function: string;
  kwargs: Record<string, unknown>;
  description?: string;
}

/**
 * Code Action - represents code execution (not used in WebShop but included for completeness).
 */
export interface ADPCodeAction {
  type: 'code_action';
  language: string;
  content: string;
  explanation?: string;
}

/**
 * Message Action - represents natural language communication.
 */
export interface ADPMessageAction {
  type: 'message_action';
  content: string;
  role?: 'assistant' | 'user';
}

/**
 * Union of all action types.
 */
export type ADPAction = ADPApiAction | ADPCodeAction | ADPMessageAction;

/**
 * Text Observation - text-based feedback from the environment.
 */
export interface ADPTextObservation {
  type: 'text_observation';
  source: 'user' | 'environment';
  content: string;
}

/**
 * Web Observation - webpage state (for web navigation tasks).
 */
export interface ADPWebObservation {
  type: 'web_observation';
  url?: string;
  html?: string;
  accessibilityTree?: string;
  screenshot?: string;
  viewport?: { width: number; height: number };
}

/**
 * Union of all observation types.
 */
export type ADPObservation = ADPTextObservation | ADPWebObservation;

/**
 * A single step in a trajectory (action-observation pair).
 */
export interface ADPStep {
  stepIndex: number;
  action: ADPAction;
  observation: ADPObservation;
  reasoning?: string;
}

/**
 * Task metadata for the trajectory.
 */
export interface ADPTaskDetails {
  taskId: string;
  instruction: string;
  domain?: string;
  targetAttributes?: Record<string, unknown>;
}

/**
 * Evaluation metadata for the trajectory.
 */
export interface ADPEvaluation {
  success: boolean;
  reward: number;
  errorMessage?: string;
}

/**
 * Complete ADP Trajectory.
 */
export interface ADPTrajectory {
  id: string;
  version: '1.0';
  timestamp: string;
  content: (ADPAction | ADPObservation)[];
  details: {
    task: ADPTaskDetails;
    evaluation: ADPEvaluation;
    modelId?: string;
    stepCount: number;
  };
}

/**
 * Collection of trajectories for export.
 */
export interface ADPDataset {
  name: string;
  description: string;
  version: string;
  trajectories: ADPTrajectory[];
  metadata: {
    domain: string;
    generatedAt: string;
    totalTrajectories: number;
    successRate: number;
    averageReward: number;
  };
}
