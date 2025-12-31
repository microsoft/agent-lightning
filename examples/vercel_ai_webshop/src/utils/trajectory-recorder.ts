/**
 * Trajectory Recorder
 *
 * Records agent steps and converts them to ADP format.
 * This is the bridge between the AI SDK's step results and the standardized
 * Agent Data Protocol format for fine-tuning.
 */

import { WebShopTask } from '@/environment/types';
import {
  ADPAction,
  ADPApiAction,
  ADPDataset,
  ADPMessageAction,
  ADPObservation,
  ADPStep,
  ADPTextObservation,
  ADPTrajectory,
} from './adp-types';

/**
 * Recorded step with both action and observation.
 */
interface RecordedStep {
  stepIndex: number;
  toolCalls: Array<{
    toolName: string;
    args: Record<string, unknown>;
  }>;
  toolResults: Array<{
    toolName: string;
    result: unknown;
  }>;
  text: string;
  reasoning?: string;
}

/**
 * Options for the trajectory recorder.
 */
export interface TrajectoryRecorderOptions {
  task: WebShopTask;
  modelId?: string;
}

/**
 * Records agent steps and converts to ADP format.
 */
export class TrajectoryRecorder {
  private steps: RecordedStep[] = [];
  private task: WebShopTask;
  private modelId: string;
  private startTime: Date;

  constructor(options: TrajectoryRecorderOptions) {
    this.task = options.task;
    this.modelId = options.modelId || 'unknown';
    this.startTime = new Date();
  }

  /**
   * Record a step from tool calls and results.
   */
  recordStep(step: {
    toolCalls: Array<{ toolName: string; args: Record<string, unknown> }>;
    toolResults: Array<{ toolName: string; result: unknown }>;
    text: string;
    reasoning?: string;
  }): void {
    this.steps.push({
      stepIndex: this.steps.length,
      toolCalls: step.toolCalls,
      toolResults: step.toolResults,
      text: step.text,
      reasoning: step.reasoning,
    });
  }

  /**
   * Convert recorded steps to ADP format.
   */
  toADPTrajectory(evaluation: {
    success: boolean;
    reward: number;
  }): ADPTrajectory {
    const content: (ADPAction | ADPObservation)[] = [];

    // Add initial task instruction as user observation
    const initialObservation: ADPTextObservation = {
      type: 'text_observation',
      source: 'user',
      content: this.task.instruction,
    };
    content.push(initialObservation);

    // Convert each step to action-observation pairs
    for (const step of this.steps) {
      // Add reasoning/text as message action if present
      if (step.text) {
        const messageAction: ADPMessageAction = {
          type: 'message_action',
          content: step.text,
          role: 'assistant',
        };
        content.push(messageAction);
      }

      // Add each tool call as an API action
      for (const toolCall of step.toolCalls) {
        const apiAction: ADPApiAction = {
          type: 'api_action',
          function: toolCall.toolName,
          kwargs: toolCall.args,
        };
        content.push(apiAction);
      }

      // Add tool results as environment observations
      for (const toolResult of step.toolResults) {
        const observation: ADPTextObservation = {
          type: 'text_observation',
          source: 'environment',
          content:
            typeof toolResult.result === 'string'
              ? toolResult.result
              : JSON.stringify(toolResult.result, null, 2),
        };
        content.push(observation);
      }
    }

    return {
      id: `trajectory_${this.task.taskId}_${Date.now()}`,
      version: '1.0',
      timestamp: this.startTime.toISOString(),
      content,
      details: {
        task: {
          taskId: this.task.taskId,
          instruction: this.task.instruction,
          domain: 'webshop',
          targetAttributes: this.task.targetAttributes,
        },
        evaluation: {
          success: evaluation.success,
          reward: evaluation.reward,
        },
        modelId: this.modelId,
        stepCount: this.steps.length,
      },
    };
  }

  /**
   * Get the recorded steps in a simplified format for debugging.
   */
  getSteps(): ADPStep[] {
    return this.steps.map((step, index) => {
      const action: ADPAction =
        step.toolCalls.length > 0
          ? {
              type: 'api_action',
              function: step.toolCalls[0].toolName,
              kwargs: step.toolCalls[0].args,
            }
          : {
              type: 'message_action',
              content: step.text,
              role: 'assistant',
            };

      const observation: ADPObservation = {
        type: 'text_observation',
        source: 'environment',
        content:
          step.toolResults.length > 0
            ? JSON.stringify(step.toolResults[0].result)
            : '',
      };

      return {
        stepIndex: index,
        action,
        observation,
        reasoning: step.reasoning,
      };
    });
  }

  /**
   * Get raw steps.
   */
  getRawSteps(): RecordedStep[] {
    return this.steps;
  }

  /**
   * Reset the recorder for a new trajectory.
   */
  reset(task: WebShopTask): void {
    this.steps = [];
    this.task = task;
    this.startTime = new Date();
  }
}

/**
 * Create an ADP dataset from multiple trajectories.
 */
export function createADPDataset(
  trajectories: ADPTrajectory[],
  options: {
    name: string;
    description: string;
    version?: string;
  }
): ADPDataset {
  const successfulTrajectories = trajectories.filter(
    t => t.details.evaluation.success
  );
  const totalReward = trajectories.reduce(
    (sum, t) => sum + t.details.evaluation.reward,
    0
  );

  return {
    name: options.name,
    description: options.description,
    version: options.version || '1.0',
    trajectories,
    metadata: {
      domain: 'webshop',
      generatedAt: new Date().toISOString(),
      totalTrajectories: trajectories.length,
      successRate:
        trajectories.length > 0
          ? successfulTrajectories.length / trajectories.length
          : 0,
      averageReward:
        trajectories.length > 0 ? totalReward / trajectories.length : 0,
    },
  };
}
