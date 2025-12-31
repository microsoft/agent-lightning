import { createAgentUIStreamResponse } from 'ai';
import { createWebShopAgent } from '@/agent/webshop-agent';
import {
  createSession,
  getSession,
  getRecorder,
  getTask,
  getEnvironment,
} from '@/environment/session-manager';
import { getTaskById, SAMPLE_TASKS } from '@/data/sample-tasks';
import { WebShopTask } from '@/environment/types';

export const maxDuration = 60;

/**
 * POST endpoint to run the agent with a task.
 *
 * Note: This UI endpoint is for debugging and testing the agent only.
 * For training, use the headless runner (scripts/headless-runner.ts) which
 * integrates with the Agent Lightning Store for rollout coordination.
 */
export async function POST(request: Request) {
  const body = await request.json();
  const {
    messages,
    taskId,
    customInstruction,
    sessionId: existingSessionId,
  } = body;

  // Determine the task
  let task: WebShopTask;
  if (taskId) {
    const foundTask = getTaskById(taskId);
    if (!foundTask) {
      return new Response(JSON.stringify({ error: 'Task not found' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    task = foundTask;
  } else if (customInstruction) {
    task = {
      taskId: `custom_${Date.now()}`,
      instruction: customInstruction,
      targetAttributes: {},
    };
  } else {
    // Default to first task
    task = SAMPLE_TASKS[0];
  }

  // Determine model ID
  const modelId = process.env.WEBSHOP_MODEL ?? 'gpt-4o-mini';

  // Create or reuse session
  let sessionId: string;
  if (existingSessionId && getSession(existingSessionId)) {
    sessionId = existingSessionId;
  } else {
    sessionId = createSession(task, modelId);
  }

  // Create agent with session-bound tools
  const agent = createWebShopAgent(sessionId, { modelId });

  // Return streaming response with session ID in headers
  const response = await createAgentUIStreamResponse({
    agent,
    messages,
  });

  // Add session ID to response headers for client tracking
  const headers = new Headers(response.headers);
  headers.set('X-Session-Id', sessionId);

  return new Response(response.body, {
    status: response.status,
    headers,
  });
}

/**
 * GET endpoint to retrieve trajectory data for export.
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');

  if (!sessionId) {
    return new Response(JSON.stringify({ error: 'Session ID required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const recorder = getRecorder(sessionId);
  const task = getTask(sessionId);
  const env = getEnvironment(sessionId);

  if (!recorder || !task) {
    return new Response(JSON.stringify({ error: 'Session not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // Get state from server environment
  const state = env?.getState();

  // Determine if task completed successfully
  const success = state?.done ?? false;
  const reward = state?.reward ?? 0;

  const trajectory = recorder.toADPTrajectory({ success, reward });

  return new Response(JSON.stringify(trajectory), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}
