/**
 * WebShop Server Environment
 *
 * HTTP adapter for connecting to the WebShop Python server.
 *
 * Run the server:
 *   cd server && source activate.sh && python webshop_server.py
 */

import {
  ActionResult,
  WebShopEnvironment,
  WebShopState,
  WebShopTask,
} from './types';

type JsonRecord = Record<string, unknown>;

export interface WebShopServerEnvOptions {
  baseUrl: string;

  /**
   * Try multiple API prefixes to handle different server configurations.
   */
  candidateApiPrefixes?: string[];
  timeoutMs?: number;
}

export class WebShopServerEnv implements WebShopEnvironment {
  private baseUrl: string;
  private candidateApiPrefixes: string[];
  private timeoutMs: number;

  private state: WebShopState = {
    step: 0,
    observation: '',
    done: false,
    reward: 0,
  };

  constructor(options: WebShopServerEnvOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.candidateApiPrefixes = options.candidateApiPrefixes ?? [''];
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  getState(): WebShopState {
    return this.state;
  }

  async reset(options?: { task?: WebShopTask }): Promise<ActionResult> {
    const task = options?.task;

    // Send task info to server - unknown fields are usually ignored
    const payload: JsonRecord = {
      session_id: this.state.sessionId,
      goal_id: task?.goalId,
      instruction: task?.instruction,
      task_id: task?.taskId,
    };

    const data = await this.postJsonWithFallback(['/reset'], payload);

    const observation =
      this.pickString(data, ['observation', 'obs', 'text']) ?? '';
    const done = this.pickBool(data, ['done', 'terminal']) ?? false;
    const reward = this.pickNumber(data, ['reward']) ?? 0;
    const sessionId =
      this.pickString(data, ['session_id', 'sessionId', 'sid']) ??
      this.state.sessionId;

    this.state = {
      sessionId,
      step: 0,
      observation,
      done,
      reward,
      info: (data as JsonRecord).info,
      lastAction: undefined,
    };

    return {
      success: true,
      observation,
      done,
      reward,
      info: (data as JsonRecord).info,
      state: this.state,
    };
  }

  async step(action: string): Promise<ActionResult> {
    const payload: JsonRecord = {
      session_id: this.state.sessionId,
      action,
    };

    const data = await this.postJsonWithFallback(['/step'], payload);

    const observation =
      this.pickString(data, ['observation', 'obs', 'text']) ?? '';
    const done = this.pickBool(data, ['done', 'terminal']) ?? false;
    const reward = this.pickNumber(data, ['reward']) ?? 0;
    const sessionId =
      this.pickString(data, ['session_id', 'sessionId', 'sid']) ??
      this.state.sessionId;

    this.state = {
      sessionId,
      step: this.state.step + 1,
      observation,
      done,
      reward,
      info: (data as JsonRecord).info,
      lastAction: action,
    };

    return {
      success: true,
      observation,
      done,
      reward,
      info: (data as JsonRecord).info,
      state: this.state,
    };
  }

  async search(query: string): Promise<ActionResult> {
    return this.step(`search[${query}]`);
  }

  async click(element: string): Promise<ActionResult> {
    return this.step(`click[${element}]`);
  }

  async buy(): Promise<ActionResult> {
    // In canonical WebShop, buying is just a click
    return this.click('Buy Now');
  }

  // -----------------------
  // HTTP plumbing
  // -----------------------

  private async postJsonWithFallback(
    paths: string[],
    body: JsonRecord
  ): Promise<unknown> {
    let lastErr: unknown;

    for (const prefix of this.candidateApiPrefixes) {
      for (const p of paths) {
        const url = `${this.baseUrl}${prefix}${p}`;

        try {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), this.timeoutMs);

          const res = await fetch(url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
          }).finally(() => clearTimeout(timer));

          if (res.status === 404) {
            // Try next prefix/path candidate
            continue;
          }

          if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(
              `WebShop server error ${res.status} on ${url}: ${text.slice(0, 500)}`
            );
          }

          const contentType = res.headers.get('content-type') || '';
          if (contentType.includes('application/json')) {
            return await res.json();
          }

          // Some servers return plain text; wrap it
          const text = await res.text();
          return { observation: text };
        } catch (err) {
          lastErr = err;
        }
      }
    }

    const isTimeout =
      lastErr instanceof Error && lastErr.name === 'AbortError';
    const hint = isTimeout
      ? `\n\nMake sure the WebShop Docker container is running:\n  docker run --rm -p 3000:3000 ainikolai/webshop:latest "0.0.0.0"`
      : '';

    throw new Error(
      `Could not reach WebShop server at ${this.baseUrl}. ` +
        `Tried prefixes=${JSON.stringify(this.candidateApiPrefixes)}, ` +
        `paths=${JSON.stringify(paths)}. ` +
        `Last error: ${lastErr instanceof Error ? lastErr.message : String(lastErr)}${hint}`
    );
  }

  private pickString(obj: unknown, keys: string[]): string | undefined {
    if (!obj || typeof obj !== 'object') return undefined;
    const o = obj as Record<string, unknown>;
    for (const k of keys) {
      const v = o[k];
      if (typeof v === 'string') return v;
    }
    return undefined;
  }

  private pickBool(obj: unknown, keys: string[]): boolean | undefined {
    if (!obj || typeof obj !== 'object') return undefined;
    const o = obj as Record<string, unknown>;
    for (const k of keys) {
      const v = o[k];
      if (typeof v === 'boolean') return v;
    }
    return undefined;
  }

  private pickNumber(obj: unknown, keys: string[]): number | undefined {
    if (!obj || typeof obj !== 'object') return undefined;
    const o = obj as Record<string, unknown>;
    for (const k of keys) {
      const v = o[k];
      if (typeof v === 'number') return v;
    }
    return undefined;
  }
}
