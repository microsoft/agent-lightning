'use client';

import { SAMPLE_TASKS } from '@/data/sample-tasks';
import { WebShopTask } from '@/environment/types';

interface TaskSelectorProps {
  value: string;
  onChange: (taskId: string) => void;
  disabled?: boolean;
}

export default function TaskSelector({
  value,
  onChange,
  disabled = false,
}: TaskSelectorProps) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="task-select" className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Select Task
      </label>
      <select
        id="task-select"
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
        className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {SAMPLE_TASKS.map(task => (
          <option key={task.taskId} value={task.taskId}>
            {task.taskId}: {task.instruction.slice(0, 50)}...
          </option>
        ))}
      </select>

      {/* Show task details */}
      <TaskDetails taskId={value} />
    </div>
  );
}

function TaskDetails({ taskId }: { taskId: string }) {
  const task = SAMPLE_TASKS.find(t => t.taskId === taskId);
  if (!task) return null;

  return (
    <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-md text-sm">
      <p className="font-medium text-gray-900 dark:text-gray-100 mb-2">
        {task.instruction}
      </p>
      <div className="flex flex-wrap gap-2">
        {Object.entries(task.targetAttributes).map(([key, value]) => (
          <span
            key={key}
            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
          >
            {key}: {value}
          </span>
        ))}
      </div>
    </div>
  );
}
