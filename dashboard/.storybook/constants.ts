// Copyright (c) Microsoft. All rights reserved.

// Centralized constants that keep Storybook fixtures deterministic so Chromatic
// snapshots do not drift when the build environment changes.
export const STORY_DATE_NOW_MS = 1762775145209;
export const STORY_DATE_NOW_SECONDS = Math.floor(STORY_DATE_NOW_MS / 1000);

// Use a fixed origin so any code that would normally read window.location.*
// in the app can rely on the same value from Storybook fixtures. Prefer HTTPS
// so Chromatic (which is served over HTTPS) avoids mixed-content fetch errors.
export const STORY_BASE_URL = 'https://storybook.agentlightning.invalid';
export const STORY_LOCATION_HREF = `${STORY_BASE_URL}/storybook`;
