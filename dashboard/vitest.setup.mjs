// Copyright (c) Microsoft. All rights reserved.

import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import '@testing-library/jest-dom/vitest';

import { vi } from 'vitest';

const serverHost = process.env.VITEST_PYTHON_HOST ?? '127.0.0.1';
const serverPort = Number.parseInt(process.env.VITEST_PYTHON_PORT ?? '8765', 10);
const serverUrl =
  (process.env.VITEST_PYTHON_URL && process.env.VITEST_PYTHON_URL.trim().replace(/\/$/, '')) ??
  `http://${serverHost}:${Number.isFinite(serverPort) ? serverPort : 8765}`;
const serverHealthUrl = `${serverUrl}/v1/agl/health`;

const globalServerState = globalThis;

const locatePythonServerPaths = () => {
  const candidates = [];
  const cwd = process.cwd();
  candidates.push({ workspace: cwd, repo: path.resolve(cwd, '..') });

  if (
    typeof import.meta !== 'undefined' &&
    typeof import.meta.url === 'string' &&
    import.meta.url.startsWith('file:')
  ) {
    const moduleDir = path.dirname(fileURLToPath(import.meta.url));
    candidates.push({ workspace: moduleDir, repo: path.resolve(moduleDir, '..') });
  }

  for (const candidate of candidates) {
    const serverScript = path.resolve(candidate.workspace, 'test-utils/python-server.py');
    if (fs.existsSync(serverScript)) {
      return { repoRoot: candidate.repo, serverScript };
    }
  }

  throw new Error('Unable to locate dashboard/test-utils/python-server.py from Vitest setup.');
};

const waitForHealth = async (timeoutMs) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const controller = new AbortController();
      const abortTimer = setTimeout(() => controller.abort(), 1000);
      const response = await fetch(serverHealthUrl, { signal: controller.signal });
      clearTimeout(abortTimer);
      if (response.ok) {
        return true;
      }
    } catch {
      // Retry until timeout.
    }
    await sleep(200);
  }
  return false;
};

const resolvePythonRunners = (repoRoot) => {
  const runners = [];
  const seenCommands = new Set();

  const pushRunner = (command, args, description, extraEnv = {}) => {
    if (seenCommands.has(command)) {
      return;
    }
    seenCommands.add(command);
    runners.push({ command, args, description, env: extraEnv });
  };

  const explicit = process.env.VITEST_PYTHON_BIN?.trim();
  if (explicit) {
    pushRunner(explicit, [], explicit);
  }

  const hasUv = (() => {
    try {
      const result = spawnSync('uv', ['--version'], { stdio: 'ignore' });
      return result.status === 0;
    } catch {
      return false;
    }
  })();

  if (hasUv) {
    const uvCacheDir = path.resolve(repoRoot, '.uv-cache');
    pushRunner('uv', ['run', '--no-sync'], 'uv run --no-sync', { UV_CACHE_DIR: uvCacheDir });
  }

  const candidates = [process.env.PYTHON, process.env.PYTHON3, 'python3', 'python'].filter(
    (candidate) => typeof candidate === 'string' && candidate.length > 0,
  );

  for (const candidate of candidates) {
    if (seenCommands.has(candidate)) {
      continue;
    }
    try {
      const result = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
      if (result.status === 0) {
        pushRunner(candidate, [], candidate);
      }
    } catch {
      // try next candidate
    }
  }

  if (runners.length === 0) {
    throw new Error('Unable to locate a Python interpreter or uv CLI for launching the LightningStore server.');
  }

  return runners;
};

const ensurePythonServer = async () => {
  if (await waitForHealth(1000)) {
    return;
  }

  const { repoRoot, serverScript } = locatePythonServerPaths();
  const runners = resolvePythonRunners(repoRoot);
  const attemptErrors = [];

  const collectOutput = (stream, buffer) => {
    if (!stream) {
      return;
    }
    stream.on('data', (chunk) => {
      buffer.push(chunk.toString());
      if (process.env.VITEST_DEBUG_PYTHON_SERVER) {
        process.stderr.write(chunk);
      }
    });
  };

  for (const runner of runners) {
    const child = spawn(runner.command, [...runner.args, serverScript], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        ...runner.env,
      },
    });

    const recordedStdout = [];
    const recordedStderr = [];
    collectOutput(child.stdout, recordedStdout);
    collectOutput(child.stderr, recordedStderr);

    let exitPayload = null;
    const exitPromise = new Promise((resolve) => {
      child.on('exit', (code, signal) => {
        exitPayload = { code, signal };
        resolve();
      });
    });

    const ready = await waitForHealth(10000);
    if (ready) {
      globalServerState.__pythonServerProcess = child;
      globalServerState.__pythonServerStartedHere = true;
      globalServerState.__pythonRunnerDescription = runner.description;

      const terminate = () => {
        if (!globalServerState.__pythonServerStartedHere) {
          return;
        }
        const proc = globalServerState.__pythonServerProcess;
        if (proc && proc.exitCode == null) {
          proc.kill('SIGTERM');
        }
      };

      if (!globalServerState.__pythonServerCleanupRegistered) {
        process.once('exit', terminate);
        process.once('SIGINT', () => {
          terminate();
          process.exit(130);
        });
        process.once('SIGTERM', () => {
          terminate();
          process.exit(143);
        });
        globalServerState.__pythonServerCleanupRegistered = true;
      }

      return;
    }

    const reused = await waitForHealth(3000);
    if (reused) {
      if (child.exitCode == null) {
        child.kill('SIGTERM');
        await exitPromise;
      }
      globalServerState.__pythonServerStartedHere = false;
      return;
    }

    if (child.exitCode == null) {
      child.kill('SIGTERM');
      await exitPromise;
    } else {
      await exitPromise;
    }

    const extra = exitPayload ? ` (code: ${exitPayload.code ?? 'null'}, signal: ${exitPayload.signal ?? 'null'})` : '';
    const stderrLog = recordedStderr.join('').trim();
    const stdoutLog = recordedStdout.join('').trim();
    const message = [
      `[${runner.description}] Failed to start LightningStore server at ${serverUrl}${extra}.`,
      stderrLog.length > 0 ? `stderr:\n${stderrLog}` : '',
      stdoutLog.length > 0 ? `stdout:\n${stdoutLog}` : '',
    ]
      .filter(Boolean)
      .join('\n\n');

    attemptErrors.push(message);
  }

  throw new Error(attemptErrors.join('\n\n'));
};

if (!globalServerState.__pythonServerPromise) {
  globalServerState.__pythonServerPromise = ensurePythonServer().catch((error) => {
    globalServerState.__pythonServerStartedHere = false;
    throw error;
  });
}

await globalServerState.__pythonServerPromise;

const { getComputedStyle } = window;
window.getComputedStyle = (elt) => getComputedStyle(elt);
window.HTMLElement.prototype.scrollIntoView = () => {};

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = ResizeObserver;
