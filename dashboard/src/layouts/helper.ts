// Copyright (c) Microsoft. All rights reserved.

function parseCssNumber(value: string | null | undefined): number {
  if (!value) {
    return 0;
  }

  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getElementWidth(selectors: string | string[]): number {
  if (typeof window === 'undefined') {
    return 0;
  }

  const selectorList = Array.isArray(selectors) ? selectors : [selectors];
  for (const selector of selectorList) {
    const element = window.document.querySelector<HTMLElement>(selector);
    if (element) {
      return element.getBoundingClientRect().width || 0;
    }
  }

  return 0;
}

function getAppShellContentWidth(): number {
  if (typeof window === 'undefined') {
    return 0;
  }

  const selectors = ['.mantine-AppShell-main', '[data-mantine-component="AppShellMain"]'];
  return getElementWidth(selectors);
}

export function getAppShellOffsets(): number {
  if (typeof window === 'undefined' || !window.document?.documentElement) {
    return 0;
  }

  const rootStyles = window.getComputedStyle(window.document.documentElement);

  const navbarOffset =
    parseCssNumber(rootStyles.getPropertyValue('--app-shell-navbar-offset')) ||
    getElementWidth(['.mantine-AppShell-navbar', '[data-mantine-component="AppShellNavbar"]']);

  const asideOffset =
    parseCssNumber(rootStyles.getPropertyValue('--app-shell-aside-offset')) ||
    getElementWidth(['.mantine-AppShell-aside', '[data-mantine-component="AppShellAside"]']);

  const paddingVar = parseCssNumber(rootStyles.getPropertyValue('--app-shell-padding'));
  let paddingTotal = paddingVar ? paddingVar * 2 : 0;

  if (paddingTotal === 0) {
    const selectors = ['.mantine-AppShell-main', '[data-mantine-component="AppShellMain"]'];
    for (const selector of selectors) {
      const main = window.document.querySelector<HTMLElement>(selector);
      if (!main) {
        continue;
      }
      const mainStyles = window.getComputedStyle(main);
      const computedPadding = parseCssNumber(mainStyles.paddingLeft) + parseCssNumber(mainStyles.paddingRight);
      if (computedPadding > 0) {
        paddingTotal = computedPadding;
        break;
      }
    }
  }

  return navbarOffset + asideOffset + paddingTotal;
}

export function getLayoutAwareWidth(containerWidth: number, viewportWidth: number): number {
  const appShellContentWidth = getAppShellContentWidth();
  const layoutOffsets = getAppShellOffsets();
  const viewportAvailable = viewportWidth && viewportWidth > 0 ? Math.max(viewportWidth - layoutOffsets, 0) : undefined;
  const effectiveAvailable = appShellContentWidth > 0 ? appShellContentWidth : viewportAvailable;

  if (!containerWidth && effectiveAvailable) {
    return effectiveAvailable;
  }

  if (!effectiveAvailable) {
    return containerWidth;
  }

  return Math.min(containerWidth, effectiveAvailable);
}
