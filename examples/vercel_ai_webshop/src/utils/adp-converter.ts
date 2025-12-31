/**
 * ADP Format Converters
 *
 * Converts ADP trajectories to various fine-tuning formats:
 * - OpenAI fine-tuning format (messages with tool_calls)
 * - Simple chat format (for models without native tool support)
 * - JSON Lines format for batch processing
 */

import { ADPDataset, ADPTrajectory } from './adp-types';

/**
 * OpenAI message format for fine-tuning.
 */
interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
}

/**
 * OpenAI fine-tuning example format.
 */
interface OpenAIFineTuneExample {
  messages: OpenAIMessage[];
}

/**
 * Convert an ADP trajectory to OpenAI fine-tuning format.
 */
export function toOpenAIFormat(
  trajectory: ADPTrajectory
): OpenAIFineTuneExample {
  const messages: OpenAIMessage[] = [];
  let toolCallCounter = 0;

  // Add system message
  messages.push({
    role: 'system',
    content: `You are a shopping assistant that helps users find and purchase products. Navigate the store by searching for products, clicking on items, selecting options, and completing purchases.`,
  });

  // Process trajectory content
  for (let i = 0; i < trajectory.content.length; i++) {
    const item = trajectory.content[i];

    if (item.type === 'text_observation') {
      if (item.source === 'user') {
        // User instruction
        messages.push({
          role: 'user',
          content: item.content,
        });
      } else {
        // Environment observation - this should follow a tool call
        // Find the previous tool call to get its ID
        const lastAssistantMsg = [...messages]
          .reverse()
          .find(m => m.role === 'assistant' && m.tool_calls);
        if (lastAssistantMsg?.tool_calls?.[0]) {
          messages.push({
            role: 'tool',
            content: item.content,
            tool_call_id: lastAssistantMsg.tool_calls[0].id,
          });
        }
      }
    } else if (item.type === 'message_action') {
      // Assistant message (reasoning or response)
      messages.push({
        role: 'assistant',
        content: item.content,
      });
    } else if (item.type === 'api_action') {
      // Tool call
      const toolCallId = `call_${toolCallCounter++}`;
      messages.push({
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: toolCallId,
            type: 'function',
            function: {
              name: item.function,
              arguments: JSON.stringify(item.kwargs),
            },
          },
        ],
      });
    }
  }

  return { messages };
}

/**
 * Convert an ADP trajectory to simple chat format (no tool_calls).
 * Tool calls are represented as text in assistant messages.
 */
export function toSimpleChatFormat(trajectory: ADPTrajectory): {
  messages: Array<{ role: string; content: string }>;
} {
  const messages: Array<{ role: string; content: string }> = [];

  // Add system message
  messages.push({
    role: 'system',
    content: `You are a shopping assistant. To interact with the store, output actions in this format:
ACTION: search(query="your search query")
ACTION: click(element="element to click")
ACTION: buy()

After each action, you will receive an observation showing the result.`,
  });

  // Process trajectory content
  for (const item of trajectory.content) {
    if (item.type === 'text_observation') {
      const role = item.source === 'user' ? 'user' : 'assistant';
      const prefix = item.source === 'environment' ? 'OBSERVATION: ' : '';
      messages.push({
        role,
        content: prefix + item.content,
      });
    } else if (item.type === 'message_action') {
      messages.push({
        role: 'assistant',
        content: item.content,
      });
    } else if (item.type === 'api_action') {
      const argsStr = Object.entries(item.kwargs)
        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
        .join(', ');
      messages.push({
        role: 'assistant',
        content: `ACTION: ${item.function}(${argsStr})`,
      });
    }
  }

  return { messages };
}

/**
 * Convert a dataset to JSON Lines format (one JSON per line).
 */
export function toJSONLines(
  dataset: ADPDataset,
  format: 'openai' | 'simple' = 'openai'
): string {
  const convertFn = format === 'openai' ? toOpenAIFormat : toSimpleChatFormat;

  return dataset.trajectories
    .map(trajectory => JSON.stringify(convertFn(trajectory)))
    .join('\n');
}

/**
 * Export a dataset to a file-ready format.
 */
export function exportDataset(
  dataset: ADPDataset,
  options: {
    format: 'openai' | 'simple' | 'adp';
    pretty?: boolean;
  }
): string {
  const { format, pretty = false } = options;

  if (format === 'adp') {
    return pretty ? JSON.stringify(dataset, null, 2) : JSON.stringify(dataset);
  }

  // For openai and simple formats, export as JSONL
  return toJSONLines(dataset, format);
}

/**
 * Filter trajectories by success or reward threshold.
 */
export function filterTrajectories(
  dataset: ADPDataset,
  filter: {
    successOnly?: boolean;
    minReward?: number;
    maxSteps?: number;
  }
): ADPDataset {
  let filtered = dataset.trajectories;

  if (filter.successOnly) {
    filtered = filtered.filter(t => t.details.evaluation.success);
  }

  if (filter.minReward !== undefined) {
    filtered = filtered.filter(
      t => t.details.evaluation.reward >= filter.minReward!
    );
  }

  if (filter.maxSteps !== undefined) {
    filtered = filtered.filter(t => t.details.stepCount <= filter.maxSteps!);
  }

  const successfulTrajectories = filtered.filter(
    t => t.details.evaluation.success
  );
  const totalReward = filtered.reduce(
    (sum, t) => sum + t.details.evaluation.reward,
    0
  );

  return {
    ...dataset,
    trajectories: filtered,
    metadata: {
      ...dataset.metadata,
      totalTrajectories: filtered.length,
      successRate:
        filtered.length > 0
          ? successfulTrajectories.length / filtered.length
          : 0,
      averageReward: filtered.length > 0 ? totalReward / filtered.length : 0,
    },
  };
}
