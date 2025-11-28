// Copyright (c) Microsoft. All rights reserved.

/**
 * Derive a human readable descriptor for an RTK Query-style error object.
 * Normalises common timeout representations to "Timeout" so the UI can surface a friendly label.
 */
export function getErrorDescriptor(error: unknown): string | null {
  if (!error || typeof error !== 'object') {
    return null;
  }

  const record = error as Record<string, unknown>;
  const statusValue = record.status;
  const statusString = typeof statusValue === 'number' || typeof statusValue === 'string' ? String(statusValue) : null;

  const data = record.data;
  let detail: string | null = null;
  if (data && typeof data === 'object') {
    const dataRecord = data as Record<string, unknown>;
    const rawDetail = dataRecord.detail ?? dataRecord.message;
    if (typeof rawDetail === 'string') {
      detail = rawDetail;
    }
  }

  const messageValue = record.message;
  const errorValue = record.error;
  const nestedErrorMessage =
    errorValue && typeof errorValue === 'object' && 'message' in (errorValue as Record<string, unknown>)
      ? (errorValue as Record<string, unknown>).message
      : null;

  const directMessage =
    typeof messageValue === 'string'
      ? messageValue
      : typeof nestedErrorMessage === 'string'
        ? nestedErrorMessage
        : typeof errorValue === 'string'
          ? errorValue
          : null;

  const looksLikeTimeout = (value: string | null) =>
    typeof value === 'string' && value.toLowerCase().includes('timeout');

  if (statusString === '408' || statusString === '504') {
    return 'Timeout';
  }
  if (looksLikeTimeout(detail)) {
    return 'Timeout';
  }
  if (looksLikeTimeout(directMessage)) {
    return 'Timeout';
  }
  if (detail && detail.trim().length > 0) {
    return detail;
  }
  if (directMessage && directMessage.trim().length > 0) {
    return directMessage;
  }
  if (statusString) {
    return `status: ${statusString}`;
  }
  return null;
}
