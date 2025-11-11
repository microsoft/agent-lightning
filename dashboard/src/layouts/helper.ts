// Copyright (c) Microsoft. All rights reserved.

function parseCssNumber(value: string | null | undefined): number {
  if (!value) {
    return 0;
  }

  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function getAppShellOffsets(): number {
  if (typeof window === 'undefined' || !window.document?.documentElement) {
    return 0;
  }

  const styles = window.getComputedStyle(window.document.documentElement);
  const navbarOffset = parseCssNumber(styles.getPropertyValue('--app-shell-navbar-offset'));
  const asideOffset = parseCssNumber(styles.getPropertyValue('--app-shell-aside-offset'));
  const padding = parseCssNumber(styles.getPropertyValue('--app-shell-padding'));

  return navbarOffset + asideOffset + padding * 2;
}

export function getLayoutAwareWidth(containerWidth: number, viewportWidth: number): number {
  if (!containerWidth) {
    return containerWidth;
  }

  const layoutOffsets = getAppShellOffsets();
  const viewportAvailable = viewportWidth && viewportWidth > 0 ? Math.max(viewportWidth - layoutOffsets, 0) : undefined;

  if (!viewportAvailable) {
    return containerWidth;
  }

  return Math.min(containerWidth, viewportAvailable);
}
