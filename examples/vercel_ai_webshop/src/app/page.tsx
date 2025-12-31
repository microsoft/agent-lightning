'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState, useEffect, useRef, useMemo } from 'react';
import TaskSelector from '@/components/task-selector';
import MessageRenderer from '@/components/message-renderer';
import TrajectoryExport from '@/components/trajectory-export';
import ChatInput from '@/components/chat-input';
import { SAMPLE_TASKS } from '@/data/sample-tasks';
import type { WebShopAgentUIMessage } from '@/agent/webshop-agent';

// Generate a unique session ID
function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export default function WebShopAgent() {
  const [selectedTaskId, setSelectedTaskId] = useState<string>('ws_001');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isTaskComplete, setIsTaskComplete] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: '/api/chat',
        body: { taskId: selectedTaskId, sessionId },
      }),
    [selectedTaskId, sessionId],
  );

  const { status, messages, sendMessage, setMessages } =
    useChat<WebShopAgentUIMessage>({
      transport,
    });

  // Check if task is complete (look for done flag in tool outputs)
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.role === 'assistant') {
      const hasDone = lastMessage.parts.some(part => {
        if (
          (part.type === 'tool-buy' ||
            part.type === 'tool-click' ||
            part.type === 'tool-search') &&
          part.state === 'output-available'
        ) {
          const output = part.output as
            | { state: 'loading' }
            | { state: 'ready'; done: boolean }
            | undefined;
          // Check if output is in ready state and has done=true
          return output?.state === 'ready' && output.done === true;
        }
        return false;
      });
      if (hasDone) {
        setIsTaskComplete(true);
      }
    }
  }, [messages]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Get selected task
  const selectedTask = SAMPLE_TASKS.find(t => t.taskId === selectedTaskId);

  // Handle task change - reset state
  const handleTaskChange = (taskId: string) => {
    setSelectedTaskId(taskId);
    setMessages([]);
    setSessionId(null);
    setIsTaskComplete(false);
  };

  // Handle run task - send the task instruction
  const handleRunTask = () => {
    if (!selectedTask) return;
    // Generate a new session ID when starting a task
    if (!sessionId) {
      setSessionId(generateSessionId());
    }
    sendMessage({
      text: `Task: ${selectedTask.instruction}\n\nPlease navigate the WebShop to find and purchase a product that matches these requirements. Start by searching for relevant products.`,
    });
  };

  // Handle custom input
  const handleSubmit = (text: string) => {
    sendMessage({ text });
  };

  const isRunning = status === 'streaming' || status === 'submitted';

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 mb-4">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              WebShop Agent
            </h1>
            <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200 rounded">
              Server Mode
            </span>
          </div>

          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <TaskSelector
                value={selectedTaskId}
                onChange={handleTaskChange}
                disabled={isRunning}
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={handleRunTask}
                disabled={isRunning || isTaskComplete}
                className="px-6 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isRunning ? (
                  <span className="flex items-center gap-2">
                    <svg
                      className="w-4 h-4 animate-spin"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
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
                    Running...
                  </span>
                ) : isTaskComplete ? (
                  'Task Complete'
                ) : (
                  'Run Task'
                )}
              </button>
            </div>
          </div>

          {/* Server status note */}
          <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            Connected to WebShop server at{' '}
            <code className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">
              {process.env.NEXT_PUBLIC_WEBSHOP_URL || 'http://localhost:3000'}
            </code>
          </p>
        </div>
      </header>

      {/* Messages Area */}
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              <svg
                className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
              <p className="text-lg font-medium">No messages yet</p>
              <p className="text-sm mt-1">
                Select a task and click &quot;Run Task&quot; to start the shopping agent
              </p>
              <p className="text-xs mt-4 text-gray-400 dark:text-gray-500">
                Make sure the WebShop Docker container is running:
                <br />
                <code className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded mt-1 inline-block">
                  docker run --rm -p 3000:3000 ainikolai/webshop:latest
                  &quot;0.0.0.0&quot;
                </code>
              </p>
            </div>
          ) : (
            messages.map(message => (
              <MessageRenderer key={message.id} message={message} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        {/* Trajectory Export */}
        <TrajectoryExport
          sessionId={sessionId || undefined}
          isComplete={isTaskComplete}
        />

        {/* Chat Input */}
        <div className="px-6 py-4">
          <div className="max-w-4xl mx-auto">
            <ChatInput
              status={status}
              onSubmit={handleSubmit}
              placeholder={
                isTaskComplete
                  ? 'Task complete! Select a new task to continue.'
                  : 'Type a custom instruction or click "Run Task" to start...'
              }
            />

            {/* Status indicator */}
            <div className="mt-2 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>
                Status:{' '}
                <span
                  className={`font-medium ${
                    status === 'ready'
                      ? 'text-green-600 dark:text-green-400'
                      : status === 'streaming'
                        ? 'text-blue-600 dark:text-blue-400'
                        : 'text-yellow-600 dark:text-yellow-400'
                  }`}
                >
                  {status}
                </span>
              </span>

              {sessionId && (
                <span>
                  Session:{' '}
                  <code className="font-mono">
                    {sessionId.slice(0, 20)}...
                  </code>
                </span>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
