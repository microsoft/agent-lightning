// Copyright (c) Microsoft. All rights reserved.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import test from "node:test";

type RequestRecord = {
  authorization?: string;
  body: Record<string, unknown>;
  path: string;
};

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
}

function sendJson(response: ServerResponse, status: number, body: object): void {
  response.writeHead(status, { Connection: "close", "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

function completion(message: object, finishReason: string): object {
  return {
    id: "chatcmpl-test",
    object: "chat.completion",
    created: 1,
    model: "auto",
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
}

async function runAgent(options: { expectedAnswer?: string; rewardStatus?: number } = {}): Promise<{
  chatRequests: RequestRecord[];
  code: number | null;
  rewardRequest?: RequestRecord;
  stderr: string;
}> {
  const chatRequests: RequestRecord[] = [];
  let rewardRequest: RequestRecord | undefined;
  const server = createServer(async (request, response) => {
    try {
      const body = await readJson(request);
      const record = {
        authorization: request.headers.authorization,
        body,
        path: request.url ?? "",
      };
      if (record.path === "/proxy/openai/v1/chat/completions") {
        chatRequests.push(record);
        if (chatRequests.length === 1) {
          sendJson(
            response,
            200,
            completion(
              {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call-1",
                    type: "function",
                    function: {
                      name: "arithmetic",
                      arguments: JSON.stringify({ operation: "add", left: 2, right: 3 }),
                    },
                  },
                ],
              },
              "tool_calls",
            ),
          );
        } else {
          sendJson(response, 200, completion({ role: "assistant", content: "### ANSWER: 5 ###" }, "stop"));
        }
        return;
      }
      if (record.path === "/events") {
        rewardRequest = record;
        sendJson(response, options.rewardStatus ?? 200, options.rewardStatus ? { detail: "rejected" } : {});
        return;
      }
      sendJson(response, 404, { detail: "not found" });
    } catch (error) {
      sendJson(response, 500, { detail: String(error) });
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert(address && typeof address !== "string");

  const entrypoint = fileURLToPath(new URL("../src/agent.js", import.meta.url));
  const child = spawn(process.execPath, [entrypoint], {
    env: {
      ...process.env,
      AGL_KEY: "test-key",
      AGL_OPENAI_BASE_URL: `http://127.0.0.1:${address.port}/proxy/openai/v1`,
      AGL_EVENT_URL: `http://127.0.0.1:${address.port}/events`,
      QUESTION: "What is 2 + 3?",
      ANSWER: options.expectedAnswer ?? "5",
    },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    stderr += chunk;
  });
  const timeout = setTimeout(() => child.kill(), 15_000);
  timeout.unref();
  const [code] = (await once(child, "exit")) as [number | null, NodeJS.Signals | null];
  clearTimeout(timeout);
  await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  return { chatRequests, code, rewardRequest, stderr };
}

test("uses non-streaming Chat Completions, completes a tool round trip, and reports reward", async () => {
  const result = await runAgent();

  assert.equal(result.code, 0, result.stderr);
  assert.equal(result.chatRequests.length, 2);
  for (const request of result.chatRequests) {
    assert.equal(request.path, "/proxy/openai/v1/chat/completions");
    assert.equal(request.authorization, "Bearer test-key");
    assert.notEqual(request.body.stream, true);
  }
  assert.equal(result.chatRequests[0].body.model, "auto");
  assert(Array.isArray(result.chatRequests[0].body.tools));
  const secondMessages = result.chatRequests[1].body.messages as Array<Record<string, unknown>>;
  assert(secondMessages.some((message) => message.role === "tool"));
  assert.deepEqual(result.rewardRequest, {
    authorization: "Bearer test-key",
    body: { event_type: "reward", data: { value: 1 } },
    path: "/events",
  });
});

test("returns a nonzero status when the reward endpoint rejects the event", async () => {
  const result = await runAgent({ rewardStatus: 500 });

  assert.equal(result.chatRequests.length, 2);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /reward request failed with 500/);
});

test("reports zero reward for an answer that only partially matches", async () => {
  const result = await runAgent({ expectedAnswer: "15" });

  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(result.rewardRequest?.body, { event_type: "reward", data: { value: 0 } });
});
