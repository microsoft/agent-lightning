// Copyright (c) Microsoft. All rights reserved.

import dayjs from 'dayjs';

export function toTimestamp(value: unknown): number | null {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }
  return value;
}

export function clampToNow(start: number | null, end: number | null): number | null {
  if (start == null) {
    return null;
  }

  const effectiveEnd = end ?? Date.now() / 1000;
  const diff = effectiveEnd - start;
  return diff > 0 ? diff : 0;
}

export function formatDateTime(timestamp: number | null): string {
  if (timestamp == null) {
    return '—';
  }

  return dayjs(timestamp * 1000).format('YYYY-MM-DD HH:mm:ss');
}

export function formatDateTimeWithMilliseconds(timestamp: number | null): string {
  if (timestamp == null) {
    return '—';
  }

  return dayjs(timestamp * 1000).format('YYYY-MM-DD HH:mm:ss.SSS');
}

export function formatDuration(seconds: number | null): string {
  if (seconds == null) {
    return '—';
  }

  const total = Math.floor(seconds);
  if (total <= 0) {
    return '—';
  }

  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const parts: string[] = [];

  if (hours > 0) {
    parts.push(`${hours}h`);
  }
  if (minutes > 0) {
    parts.push(`${minutes}m`);
  }
  if (secs > 0 || parts.length === 0) {
    parts.push(`${secs}s`);
  }

  return parts.join(' ');
}

export function formatRelativeTime(timestamp: number | null): string {
  if (timestamp == null) {
    return '—';
  }

  const now = Date.now() / 1000;
  const diff = Math.floor(now - timestamp);

  if (diff <= 5) {
    return 'Just now';
  }

  return `${formatDuration(diff)} ago`;
}

export function formatStatusLabel(status: string): string {
  return status
    .replace(/[_-]/g, ' ')
    .toLowerCase()
    .replace(/^\w|\s\w/g, (match) => match.toUpperCase());
}

export function formatInputPreview(input: unknown, maxLength = 35): { preview: string; full: string } {
  if (input === null || input === undefined) {
    return { preview: '—', full: '—' };
  }

  const serialized = typeof input === 'string' ? input : safeStringify(input);
  if (serialized.length <= maxLength) {
    return { preview: serialized, full: serialized };
  }

  return { preview: `${serialized.slice(0, maxLength - 3)}...`, full: serialized };
}

export function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const toCamelCaseKey = (key: string) => key.replace(/_([a-z])/g, (_match, letter: string) => letter.toUpperCase());

const toSnakeCaseKey = (key: string) =>
  key
    .replace(/([A-Z])/g, '_$1')
    .replace(/[-\s]+/g, '_')
    .toLowerCase();

export const camelCaseKeys = <T>(value: T): T => {
  if (Array.isArray(value)) {
    return value.map((item) => camelCaseKeys(item)) as unknown as T;
  }

  if (isPlainObject(value)) {
    const entries = Object.entries(value).map(([key, item]) => [toCamelCaseKey(key), camelCaseKeys(item)]);
    return Object.fromEntries(entries) as T;
  }

  return value;
};

export const snakeCaseKeys = <T>(value: T): T => {
  if (Array.isArray(value)) {
    return value.map((item) => snakeCaseKeys(item)) as unknown as T;
  }

  if (isPlainObject(value)) {
    const entries = Object.entries(value).map(([key, item]) => [toSnakeCaseKey(key), snakeCaseKeys(item)]);
    return Object.fromEntries(entries) as T;
  }

  return value;
};
