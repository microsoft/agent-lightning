// Copyright (c) Microsoft. All rights reserved.

/**
 * Sample WebShop Tasks
 *
 * A collection of sample tasks for the WebShop environment, designed to
 * test agent navigation, product matching, and option selection capabilities.
 */

import { WebShopTask } from '@/environment/types';

/**
 * Sample tasks covering different product categories and requirements.
 */
export const SAMPLE_TASKS: WebShopTask[] = [
  {
    taskId: 'ws_001',
    instruction: "I need a red cotton t-shirt for men, size large, under $30",
    targetAttributes: {
      color: 'red',
      material: 'cotton',
      size: 'L',
      priceMax: 30,
    },
  },
  {
    taskId: 'ws_002',
    instruction:
      "Find me black running shorts for men, size medium, athletic style",
    targetAttributes: {
      color: 'black',
      style: 'athletic',
      size: 'M',
    },
  },
  {
    taskId: 'ws_003',
    instruction: 'I want a cozy gray fleece hoodie for women, size small',
    targetAttributes: {
      color: 'gray',
      material: 'fleece',
      size: 'S',
    },
  },
  {
    taskId: 'ws_004',
    instruction: 'Looking for white canvas sneakers, size 9, under $50',
    targetAttributes: {
      color: 'white',
      material: 'canvas',
      size: '9',
      priceMax: 50,
    },
  },
  {
    taskId: 'ws_005',
    instruction: 'I need navy slim fit chino pants, waist 32, length 32',
    targetAttributes: {
      color: 'navy',
      style: 'slim fit',
      waist: '32',
      length: '32',
    },
  },
  {
    taskId: 'ws_006',
    instruction: 'Find black yoga leggings for women, size medium, high-waisted',
    targetAttributes: {
      color: 'black',
      style: 'leggings',
      size: 'M',
    },
  },
  {
    taskId: 'ws_007',
    instruction:
      'I want a white organic cotton blouse for women, size medium',
    targetAttributes: {
      color: 'white',
      material: 'organic cotton',
      size: 'M',
    },
  },
  {
    taskId: 'ws_008',
    instruction: 'Looking for a navy polo shirt for men, size XL, under $50',
    targetAttributes: {
      color: 'navy',
      style: 'polo',
      size: 'XL',
      priceMax: 50,
    },
  },
];

/**
 * Get a random subset of tasks.
 */
export function getRandomTasks(count: number): WebShopTask[] {
  const shuffled = [...SAMPLE_TASKS].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, shuffled.length));
}

/**
 * Get a task by ID.
 */
export function getTaskById(taskId: string): WebShopTask | undefined {
  return SAMPLE_TASKS.find(t => t.taskId === taskId);
}
