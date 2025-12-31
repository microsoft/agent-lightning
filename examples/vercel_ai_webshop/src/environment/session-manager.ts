/**
 * Session Manager
 *
 * Manages per-request WebShop server environment instances.
 * Each session has its own environment state and trajectory recorder.
 */

import { WebShopTask, WebShopEnvironment } from './types';
import { WebShopServerEnv } from './webshop-server';
import { TrajectoryRecorder } from '@/utils/trajectory-recorder';

/**
 * Agent Lightning rollout tracking information.
 */
export interface AgentLightningRolloutInfo {
  rolloutId: string;
  attemptId: string;
  mode: string;
}

interface Session {
  id: string;
  environment: WebShopServerEnv;
  task: WebShopTask;
  recorder: TrajectoryRecorder;
  createdAt: Date;
  initialized: boolean;
  alRollout?: AgentLightningRolloutInfo;
}

// In-memory session store (would be Redis in production)
const sessions = new Map<string, Session>();

// Clean up old sessions after 30 minutes
const SESSION_TTL_MS = 30 * 60 * 1000;

// Default WebShop server URL
const DEFAULT_WEBSHOP_URL =
  process.env.WEBSHOP_URL ?? 'http://localhost:3000';

/**
 * Generate a unique session ID.
 */
function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Clean up expired sessions.
 */
function cleanupExpiredSessions(): void {
  const now = Date.now();
  for (const [id, session] of sessions.entries()) {
    if (now - session.createdAt.getTime() > SESSION_TTL_MS) {
      sessions.delete(id);
    }
  }
}

/**
 * Create a new session with a WebShop server environment.
 */
export function createSession(
  task: WebShopTask,
  modelId?: string,
  webshopUrl?: string
): string {
  cleanupExpiredSessions();

  const sessionId = generateSessionId();
  const environment = new WebShopServerEnv({
    baseUrl: webshopUrl ?? DEFAULT_WEBSHOP_URL,
    timeoutMs: 30_000,
  });
  const recorder = new TrajectoryRecorder({
    task,
    modelId: modelId || 'gpt-4o-mini',
  });

  sessions.set(sessionId, {
    id: sessionId,
    environment,
    task,
    recorder,
    createdAt: new Date(),
    initialized: false,
  });

  return sessionId;
}

/**
 * Get a session by ID.
 */
export function getSession(sessionId: string): Session | undefined {
  return sessions.get(sessionId);
}

/**
 * Get the environment for a session.
 */
export function getEnvironment(
  sessionId: string
): WebShopServerEnv | undefined {
  return sessions.get(sessionId)?.environment;
}

/**
 * Get the recorder for a session.
 */
export function getRecorder(sessionId: string): TrajectoryRecorder | undefined {
  return sessions.get(sessionId)?.recorder;
}

/**
 * Get the task for a session.
 */
export function getTask(sessionId: string): WebShopTask | undefined {
  return sessions.get(sessionId)?.task;
}

/**
 * Check if session has been initialized (reset called).
 */
export function isSessionInitialized(sessionId: string): boolean {
  return sessions.get(sessionId)?.initialized ?? false;
}

/**
 * Mark session as initialized.
 */
export function markSessionInitialized(sessionId: string): void {
  const session = sessions.get(sessionId);
  if (session) {
    session.initialized = true;
  }
}

/**
 * Reset a session's environment.
 */
export async function resetSession(sessionId: string): Promise<boolean> {
  const session = sessions.get(sessionId);
  if (!session) return false;

  await session.environment.reset({ task: session.task });
  session.recorder.reset(session.task);
  session.initialized = true;
  return true;
}

/**
 * Delete a session.
 */
export function deleteSession(sessionId: string): boolean {
  return sessions.delete(sessionId);
}

/**
 * Get all active session IDs (for debugging).
 */
export function getActiveSessionIds(): string[] {
  return Array.from(sessions.keys());
}

/**
 * Get WebShop server URL from environment.
 */
export function getWebShopUrl(): string {
  return DEFAULT_WEBSHOP_URL;
}

/**
 * Get Agent Lightning rollout info for a session.
 */
export function getAgentLightningRollout(
  sessionId: string
): AgentLightningRolloutInfo | undefined {
  return sessions.get(sessionId)?.alRollout;
}

/**
 * Set Agent Lightning rollout info for a session.
 */
export function setAgentLightningRollout(
  sessionId: string,
  info: AgentLightningRolloutInfo
): void {
  const session = sessions.get(sessionId);
  if (session) {
    session.alRollout = info;
  }
}
