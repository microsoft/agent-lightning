'use client';

import { useState, FormEvent } from 'react';

interface ChatInputProps {
  status: string;
  onSubmit: (text: string) => void;
  placeholder?: string;
}

export default function ChatInput({
  status,
  onSubmit,
  placeholder = 'Type your instruction or click "Run Task" to start...',
}: ChatInputProps) {
  const [text, setText] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (text.trim() === '') return;
    onSubmit(text);
    setText('');
  };

  const isDisabled = status !== 'ready';

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={text}
        onChange={e => setText(e.target.value)}
        disabled={isDisabled}
        placeholder={placeholder}
        className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <button
        type="submit"
        disabled={isDisabled || text.trim() === ''}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Send
      </button>
    </form>
  );
}
