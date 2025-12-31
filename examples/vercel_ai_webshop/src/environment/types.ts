/**
 * WebShop Environment Types (Server-backed)
 *
 * Types for connecting to the real WebShop Docker environment.
 */

/**
 * Product attributes for task metadata and training data.
 */
export interface ProductAttributes {
  color?: string;
  size?: string;
  material?: string;
  brand?: string;
  style?: string;
  [key: string]: string | number | undefined;
}

/**
 * Task instruction for the WebShop agent.
 */
export interface WebShopTask {
  taskId: string;
  instruction: string;
  targetAttributes: ProductAttributes & {
    priceMax?: number;
    priceMin?: number;
  };

  /**
   * Optional: environment-native goal id/index for real WebShop datasets.
   */
  goalId?: string | number;
}

/**
 * Server-oriented state - only tracks what the server returns.
 */
export interface WebShopState {
  sessionId?: string;
  step: number;
  observation: string;
  done: boolean;
  reward: number;
  info?: unknown;
  lastAction?: string;
}

/**
 * Result of executing an action in the WebShop environment.
 */
export interface ActionResult {
  success: boolean;
  observation: string;
  done: boolean;
  reward?: number;
  info?: unknown;
  state: WebShopState;
}

/**
 * WebShop environment interface (async for server communication).
 */
export interface WebShopEnvironment {
  getState(): WebShopState;

  /**
   * Start a new episode/session on the server.
   */
  reset(options?: { task?: WebShopTask }): Promise<ActionResult>;

  /**
   * Low-level step - send an action string to the server.
   */
  step(action: string): Promise<ActionResult>;

  /**
   * Search for products. Maps to search[query] in WebShop.
   */
  search(query: string): Promise<ActionResult>;

  /**
   * Click on an element. Maps to click[element] in WebShop.
   */
  click(element: string): Promise<ActionResult>;

  /**
   * Convenience method for purchasing (usually click[Buy Now]).
   */
  buy(): Promise<ActionResult>;
}

/**
 * Search result item (parsed from observation text).
 */
export interface SearchResultItem {
  id: string;
  name: string;
  price: number;
  rating?: number;
  attributes: ProductAttributes;
}

/**
 * Product info (parsed from observation text).
 */
export interface Product {
  id: string;
  name: string;
  price: number;
  description: string;
  attributes: ProductAttributes;
  options: Array<{ name: string; values: string[] }>;
  rating?: number;
  reviewCount?: number;
}

/**
 * Selected options for a product purchase.
 */
export interface SelectedOptions {
  [optionName: string]: string;
}
