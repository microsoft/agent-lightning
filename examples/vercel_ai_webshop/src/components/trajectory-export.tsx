'use client';

import { useState, useEffect, useCallback } from 'react';
import { ADPTrajectory } from '@/utils/adp-types';
import { toOpenAIFormat, toSimpleChatFormat } from '@/utils/adp-converter';

type ExportFormat = 'adp' | 'openai' | 'simple';

interface TrajectoryExportProps {
  sessionId?: string;
  isComplete?: boolean;
}

export default function TrajectoryExport({
  sessionId,
  isComplete = false,
}: TrajectoryExportProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [format, setFormat] = useState<ExportFormat>('adp');
  const [trajectory, setTrajectory] = useState<ADPTrajectory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTrajectory = useCallback(async () => {
    if (!sessionId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/chat?sessionId=${sessionId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch trajectory');
      }
      const data = await response.json();
      setTrajectory(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Fetch trajectory when opened and task is complete
  useEffect(() => {
    if (isOpen && sessionId && isComplete) {
      fetchTrajectory();
    }
  }, [isOpen, sessionId, isComplete, fetchTrajectory]);

  const getFormattedOutput = () => {
    if (!trajectory) return '';

    switch (format) {
      case 'adp':
        return JSON.stringify(trajectory, null, 2);
      case 'openai':
        return JSON.stringify(toOpenAIFormat(trajectory), null, 2);
      case 'simple':
        return JSON.stringify(toSimpleChatFormat(trajectory), null, 2);
    }
  };

  const handleDownload = () => {
    const content = getFormattedOutput();
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trajectory_${trajectory?.id || 'export'}.${format === 'openai' ? 'jsonl' : 'json'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!isComplete) {
    return null;
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700">
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
      >
        <span className="font-medium text-gray-900 dark:text-gray-100">
          Export Trajectory
        </span>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Export Panel */}
      {isOpen && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800">
          {/* Format Selector */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Export Format
            </label>
            <div className="flex gap-2">
              {(['adp', 'openai', 'simple'] as ExportFormat[]).map(fmt => (
                <button
                  key={fmt}
                  onClick={() => setFormat(fmt)}
                  className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                    format === fmt
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  {fmt === 'adp' && 'ADP Format'}
                  {fmt === 'openai' && 'OpenAI Fine-tune'}
                  {fmt === 'simple' && 'Simple Chat'}
                </button>
              ))}
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-8">
              <svg className="w-6 h-6 animate-spin text-blue-600" fill="none" viewBox="0 0 24 24">
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
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded mb-4">
              <span className="text-red-600 dark:text-red-400">{error}</span>
            </div>
          )}

          {/* Preview */}
          {trajectory && !loading && (
            <>
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Preview
                  </label>
                  <button
                    onClick={handleDownload}
                    className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                    Download
                  </button>
                </div>
                <pre className="p-3 bg-gray-900 text-gray-100 rounded-lg text-xs overflow-auto max-h-64 font-mono">
                  {getFormattedOutput().slice(0, 2000)}
                  {getFormattedOutput().length > 2000 && '\n...(truncated)'}
                </pre>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-2 bg-white dark:bg-gray-700 rounded">
                  <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    {trajectory.details.stepCount}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Steps</div>
                </div>
                <div className="p-2 bg-white dark:bg-gray-700 rounded">
                  <div className={`text-lg font-bold ${
                    trajectory.details.evaluation.success
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {trajectory.details.evaluation.success ? 'Yes' : 'No'}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Success</div>
                </div>
                <div className="p-2 bg-white dark:bg-gray-700 rounded">
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                    {(trajectory.details.evaluation.reward * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Reward</div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
