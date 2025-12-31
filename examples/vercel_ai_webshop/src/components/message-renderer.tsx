'use client';

import { WebShopAgentUIMessage } from '@/agent/webshop-agent';
import type {
  SearchToolOutput,
  ClickToolOutput,
  BuyToolOutput,
} from '@/tools';

interface MessageRendererProps {
  message: WebShopAgentUIMessage;
}

export default function MessageRenderer({ message }: MessageRendererProps) {
  return (
    <div
      className={`p-4 rounded-lg ${
        message.role === 'user'
          ? 'bg-blue-50 dark:bg-blue-900/20 ml-8'
          : 'bg-white dark:bg-gray-800 mr-8 border border-gray-200 dark:border-gray-700'
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded ${
            message.role === 'user'
              ? 'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100'
              : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-100'
          }`}
        >
          {message.role === 'user' ? 'User' : 'Agent'}
        </span>
      </div>

      <div className="space-y-3">
        {message.parts.map((part, index) => (
          <MessagePart key={index} part={part} />
        ))}
      </div>
    </div>
  );
}

function MessagePart({
  part,
}: {
  part: WebShopAgentUIMessage['parts'][number];
}) {
  switch (part.type) {
    case 'text':
      return (
        <p className="text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
          {part.text}
        </p>
      );

    case 'step-start':
      return <hr className="border-gray-200 dark:border-gray-600 my-2" />;

    case 'tool-search':
      return <SearchToolView invocation={part} />;

    case 'tool-click':
      return <ClickToolView invocation={part} />;

    case 'tool-buy':
      return <BuyToolView invocation={part} />;

    default:
      // Handle unknown tool types gracefully
      return (
        <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded text-sm">
          <pre className="overflow-auto">{JSON.stringify(part, null, 2)}</pre>
        </div>
      );
  }
}

// Type helpers for tool invocations
type ToolInvocation<T extends string> = Extract<
  WebShopAgentUIMessage['parts'][number],
  { type: T }
>;

function SearchToolView({
  invocation,
}: {
  invocation: ToolInvocation<'tool-search'>;
}) {
  const input = invocation.input as { query?: string } | undefined;
  const output = invocation.output as SearchToolOutput | undefined;

  switch (invocation.state) {
    case 'input-streaming':
    case 'input-available':
      return (
        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
          <LoadingSpinner />
          <span>Searching for &quot;{input?.query || '...'}&quot;</span>
        </div>
      );

    case 'output-available':
      if (!output || output.state === 'loading') {
        return (
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <LoadingSpinner />
            <span>Searching...</span>
          </div>
        );
      }
      return (
        <ObservationView
          title={`Search: "${output.query}"`}
          observation={output.observation}
          done={output.done}
          reward={output.reward}
        />
      );

    case 'output-error':
      return <ErrorView message={invocation.errorText} />;
  }
}

function ClickToolView({
  invocation,
}: {
  invocation: ToolInvocation<'tool-click'>;
}) {
  const input = invocation.input as { element?: string } | undefined;
  const output = invocation.output as ClickToolOutput | undefined;

  switch (invocation.state) {
    case 'input-streaming':
    case 'input-available':
      return (
        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
          <LoadingSpinner />
          <span>Clicking &quot;{input?.element || '...'}&quot;</span>
        </div>
      );

    case 'output-available':
      if (!output || output.state === 'loading') {
        return (
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <LoadingSpinner />
            <span>Processing click...</span>
          </div>
        );
      }
      return (
        <ObservationView
          title={`Click: "${output.element}"`}
          observation={output.observation}
          done={output.done}
          reward={output.reward}
        />
      );

    case 'output-error':
      return <ErrorView message={invocation.errorText} />;
  }
}

function BuyToolView({
  invocation,
}: {
  invocation: ToolInvocation<'tool-buy'>;
}) {
  const output = invocation.output as BuyToolOutput | undefined;

  switch (invocation.state) {
    case 'input-streaming':
    case 'input-available':
      return (
        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
          <LoadingSpinner />
          <span>Processing purchase...</span>
        </div>
      );

    case 'output-available':
      if (!output || output.state === 'loading') {
        return (
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <LoadingSpinner />
            <span>Processing purchase...</span>
          </div>
        );
      }

      // Purchase complete - show result
      return (
        <PurchaseResultView
          observation={output.observation}
          success={output.success}
          reward={output.reward}
          done={output.done}
        />
      );

    case 'output-error':
      return <ErrorView message={invocation.errorText} />;
  }
}

// Helper components

function LoadingSpinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

function ErrorView({ message }: { message: string }) {
  return (
    <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
      <span className="text-red-600 dark:text-red-400">Error: {message}</span>
    </div>
  );
}

function ObservationView({
  title,
  observation,
  done,
  reward,
}: {
  title: string;
  observation: string;
  done: boolean;
  reward?: number;
}) {
  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          {title}
        </h4>
        <div className="flex items-center gap-2">
          {done && (
            <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200 rounded">
              Done
            </span>
          )}
          {reward !== undefined && reward > 0 && (
            <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200 rounded">
              Reward: {(reward * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>
      <pre className="text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap overflow-auto max-h-96 font-mono bg-white dark:bg-gray-900 p-3 rounded border border-gray-100 dark:border-gray-700">
        {observation}
      </pre>
    </div>
  );
}

function PurchaseResultView({
  observation,
  success,
  reward,
  done,
}: {
  observation: string;
  success: boolean;
  reward: number;
  done: boolean;
}) {
  const rewardColor =
    reward >= 0.8
      ? 'text-green-600 dark:text-green-400'
      : reward >= 0.5
        ? 'text-yellow-600 dark:text-yellow-400'
        : 'text-red-600 dark:text-red-400';

  return (
    <div
      className={`p-4 rounded-lg ${
        success && done
          ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
          : 'bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
      }`}
    >
      <div className="flex items-center gap-3 mb-3">
        {success && done ? (
          <div className="flex items-center justify-center w-10 h-10 rounded-full bg-green-100 dark:bg-green-800">
            <svg
              className="w-6 h-6 text-green-600 dark:text-green-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
        ) : (
          <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-100 dark:bg-gray-700">
            <svg
              className="w-6 h-6 text-gray-600 dark:text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"
              />
            </svg>
          </div>
        )}
        <div>
          <h4 className="font-semibold text-lg text-gray-900 dark:text-gray-100">
            {success && done ? 'Purchase Complete!' : 'Buy Action'}
          </h4>
          {done && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Task completed
            </p>
          )}
        </div>
      </div>

      {/* Reward Score */}
      {done && (
        <div className="mb-4 p-3 bg-white dark:bg-gray-800 rounded-md">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Reward Score
            </span>
            <span className={`text-2xl font-bold ${rewardColor}`}>
              {(reward * 100).toFixed(0)}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                reward >= 0.8
                  ? 'bg-green-500'
                  : reward >= 0.5
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
              }`}
              style={{ width: `${reward * 100}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Observation */}
      <pre className="text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap overflow-auto max-h-48 font-mono bg-white dark:bg-gray-900 p-3 rounded border border-gray-100 dark:border-gray-700">
        {observation}
      </pre>
    </div>
  );
}
