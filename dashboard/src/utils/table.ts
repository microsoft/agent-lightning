// Copyright (c) Microsoft. All rights reserved.

import type { DataTableColumn } from 'mantine-datatable';

/**
 * Configuration for column visibility based on screen width
 */
export type ColumnVisibilityConfig =
  | {
      minWidth: number;
      priority: number;
      fixedWidth?: never;
    }
  | {
      fixedWidth: number;
      priority: number;
      minWidth?: never;
    };

/**
 * Compare two records for sorting purposes
 * Handles null/undefined values and both string and number comparisons
 */
export function compareRecords<T, K extends keyof T>(a: T, b: T, key: K): number {
  const valueA = a[key];
  const valueB = b[key];

  if (valueA === valueB) {
    return 0;
  }

  if (valueA === null || valueA === undefined) {
    return 1;
  }

  if (valueB === null || valueB === undefined) {
    return -1;
  }

  if (typeof valueA === 'number' && typeof valueB === 'number') {
    return valueA - valueB;
  }

  return String(valueA).localeCompare(String(valueB));
}

/**
 * Create responsive columns based on container width
 * Columns with priority 0 are always shown, others are shown based on available space
 */
const FALLBACK_EM_IN_PIXELS = 16;

function getEmInPixels(): number {
  if (typeof window === 'undefined' || !window.document?.documentElement) {
    return FALLBACK_EM_IN_PIXELS;
  }

  const rootFontSize = window.getComputedStyle(window.document.documentElement).fontSize;
  const parsed = Number.parseFloat(rootFontSize);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }

  return FALLBACK_EM_IN_PIXELS;
}

function resolveWidth(config: ColumnVisibilityConfig): { widthEm: number; fixed: boolean } {
  if ('fixedWidth' in config && typeof config.fixedWidth === 'number') {
    return { widthEm: config.fixedWidth, fixed: true };
  }
  return { widthEm: config.minWidth, fixed: false };
}

export function createResponsiveColumns<T>(
  columns: DataTableColumn<T>[],
  containerWidth: number,
  columnVisibilityConfig: Record<string, ColumnVisibilityConfig>,
): DataTableColumn<T>[] {
  const measuredWidth = containerWidth ? Math.max(containerWidth, 0) : Number.POSITIVE_INFINITY;
  const emInPixels = getEmInPixels();

  const columnEntries = columns.map((column, index) => {
    const accessorKey = String(column.accessor);
    const config =
      columnVisibilityConfig[accessorKey] ??
      ({
        minWidth: 10,
        priority: 3,
      } satisfies ColumnVisibilityConfig);
    const { widthEm, fixed } = resolveWidth(config);
    return {
      column,
      index,
      accessorKey,
      ...config,
      widthEm,
      widthPx: widthEm * emInPixels,
      fixed,
    };
  });

  const sortedByPriority = columnEntries
    .slice()
    .sort((a, b) => (a.priority === b.priority ? a.index - b.index : a.priority - b.priority));

  const visibleColumnIndices = new Set<number>();
  let usedWidth = 0;

  // First pass: add all priority 0 columns
  sortedByPriority.forEach((entry) => {
    if (entry.priority === 0) {
      visibleColumnIndices.add(entry.index);
      usedWidth += entry.widthPx;
    }
  });

  // Second pass: add remaining columns that fit
  sortedByPriority.forEach((entry) => {
    if (visibleColumnIndices.has(entry.index)) {
      return;
    }
    if (usedWidth + entry.widthPx <= measuredWidth) {
      visibleColumnIndices.add(entry.index);
      usedWidth += entry.widthPx;
    }
  });

  return columnEntries.map(({ column, index, fixed, widthEm }) => {
    const columnWidth = `${widthEm}em`;
    const existingStyle = (column as any).style ?? {};
    return {
      ...column,
      width: fixed ? columnWidth : column.width,
      style: {
        ...existingStyle,
        minWidth: columnWidth,
      },
      hidden: !visibleColumnIndices.has(index),
    };
  });
}
