// Copyright (c) Microsoft. All rights reserved.

import { createOpenAI } from "@ai-sdk/openai";
import { generateText, stepCountIs, tool } from "ai";
import { pathToFileURL } from "node:url";
import { z } from "zod";

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

export function extractFinalAnswer(text: string): string {
  const trimmed = text.trim();
  const delimited = trimmed.match(/###\s*ANSWER:\s*(.+?)(?:\s*###|$)/is);
  if (delimited) {
    return delimited[1].trim();
  }
  const gsm8k = trimmed.match(/####\s*(.+)$/is);
  if (gsm8k) {
    return gsm8k[1].trim();
  }
  const numbers = trimmed.match(/[-+]?\d[\d,]*(?:\.\d+)?/g);
  return numbers?.at(-1)?.replaceAll(",", "") ?? trimmed;
}

function normalizeAnswer(text: string): string {
  return extractFinalAnswer(text).replaceAll(",", "").trim();
}

const arithmetic = tool({
  description: "Perform one arithmetic operation with two numbers.",
  inputSchema: z.object({
    operation: z.enum(["add", "subtract", "multiply", "divide"]),
    left: z.number(),
    right: z.number(),
  }),
  execute: async ({ operation, left, right }) => {
    if (operation === "add") return left + right;
    if (operation === "subtract") return left - right;
    if (operation === "multiply") return left * right;
    if (right === 0) throw new Error("division by zero");
    return left / right;
  },
});

async function postReward(eventUrl: string, apiKey: string, value: number): Promise<void> {
  const response = await fetch(eventUrl, {
    method: "POST",
    signal: AbortSignal.timeout(10_000),
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      event_type: "reward",
      data: { value },
    }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`reward request failed with ${response.status}: ${detail}`);
  }
}

export async function run(): Promise<void> {
  const apiKey = requiredEnv("AGL_KEY");
  const question = requiredEnv("QUESTION");
  const expectedAnswer = requiredEnv("ANSWER");
  const eventUrl = requiredEnv("AGL_EVENT_URL");
  const baseURL = requiredEnv("AGL_OPENAI_BASE_URL");

  const openai = createOpenAI({ apiKey, baseURL });
  const result = await generateText({
    model: openai.chat("auto"),
    system:
      "Solve the grade-school math problem. Use the arithmetic tool for calculations. " +
      "End with exactly ### ANSWER: <answer> ###.",
    prompt: question,
    tools: { arithmetic },
    stopWhen: stepCountIs(4),
    temperature: 1,
    maxOutputTokens: 1024,
  });

  const reward = normalizeAnswer(result.text) === normalizeAnswer(expectedAnswer) ? 1 : 0;
  await postReward(eventUrl, apiKey, reward);
}

const invokedPath = process.argv[1] ? pathToFileURL(process.argv[1]).href : undefined;
if (invokedPath === import.meta.url) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
