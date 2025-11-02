// Copyright (c) Microsoft. All rights reserved.

// @ts-check
import stylistic from '@stylistic/eslint-plugin';
import mantine from 'eslint-config-mantine';
import { defineConfig } from 'eslint/config';
import tseslint from 'typescript-eslint';

export default defineConfig([
  // These are arrays → safe to spread
  ...tseslint.configs.recommended,
  stylistic.configs.customize({ semi: true }),

  // mantine is often a single object → include as-is (or spread only if it's actually an array)
  ...(Array.isArray(mantine) ? mantine : [mantine]),

  // ignores go as their own entry
  { ignores: ['**/*.{mjs,cjs,js,d.ts,d.mts}'] },

  // file-specific rules
  {
    files: ['**/*.story.tsx'],
    rules: { 'no-console': 'off' },
  },

  // project/TS settings + your custom rules
  {
    languageOptions: {
      parserOptions: {
        tsconfigRootDir: process.cwd(),
        project: ['./tsconfig.json'],
      },
    },
    rules: {
      // Disabling conflict rules with prettier
      '@stylistic/brace-style': ['error', '1tbs', { allowSingleLine: false }],
      '@stylistic/no-trailing-spaces': 'error',
      '@stylistic/no-multiple-empty-lines': ['error', { max: 2, maxEOF: 1 }],
      '@stylistic/jsx-quotes': ['error', 'prefer-single'],
      '@stylistic/multiline-ternary': 'off',
      '@stylistic/arrow-parens': ['error', 'always'],
      '@stylistic/jsx-closing-bracket-location': 'off',
      '@stylistic/operator-linebreak': 'off',
      '@stylistic/jsx-newline': 'off',
      '@stylistic/jsx-one-expression-per-line': 'off',
      '@stylistic/indent': 'off',
      '@stylistic/indent-binary-ops': 'off',
    },
  },
]);
