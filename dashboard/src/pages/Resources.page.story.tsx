// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { waitFor, within } from '@testing-library/dom';
import userEvent from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';
import { Provider } from 'react-redux';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { AppAlertBanner } from '@/components/AppAlertBanner';
import { AppDrawerContainer } from '@/components/AppDrawer.component';
import { initialConfigState } from '@/features/config/slice';
import { initialResourcesUiState } from '@/features/resources/slice';
import { initialRolloutsUiState } from '@/features/rollouts/slice';
import { AppLayout } from '@/layouts/AppLayout';
import { createAppStore } from '@/store';
import type { Resources } from '@/types';
import { createResourcesHandlers } from '@/utils/mock';
import { STORY_BASE_URL, STORY_DATE_NOW_SECONDS } from '../../.storybook/constants';
import { allModes } from '../../.storybook/modes';
import { ResourcesPage } from './Resources.page';

const meta: Meta<typeof ResourcesPage> = {
  title: 'Pages/ResourcesPage',
  component: ResourcesPage,
  parameters: {
    layout: 'fullscreen',
    chromatic: {
      modes: allModes,
    },
  },
};

export default meta;

type Story = StoryObj<typeof ResourcesPage>;

const now = STORY_DATE_NOW_SECONDS;

const sampleResources: Resources[] = [
  {
    resourcesId: 'rs-a1b2c3d4e5f6',
    version: 3,
    createTime: now - 7 * 24 * 3600,
    updateTime: now - 3600,
    resources: {
      main_llm: {
        resource_type: 'llm',
        endpoint: 'https://api.openai.com/v1',
        model: 'gpt-4',
        sampling_parameters: {
          temperature: 0.7,
          max_tokens: 2048,
        },
      },
      backup_llm: {
        resource_type: 'llm',
        endpoint: 'https://api.anthropic.com/v1',
        model: 'claude-3-opus-20240229',
        sampling_parameters: {
          temperature: 0.5,
          max_tokens: 4096,
        },
      },
      greeting_template: {
        resource_type: 'prompt_template',
        template: 'Hello {name}! Welcome to {service}.',
        engine: 'f-string',
      },
    },
  },
  {
    resourcesId: 'rs-f6e5d4c3b2a1',
    version: 2,
    createTime: now - 14 * 24 * 3600,
    updateTime: now - 2 * 24 * 3600,
    resources: {
      production_llm: {
        resource_type: 'llm',
        endpoint: 'https://api.openai.com/v1',
        model: 'gpt-3.5-turbo',
        sampling_parameters: {
          temperature: 0.8,
          max_tokens: 1024,
        },
      },
      system_prompt: {
        resource_type: 'prompt_template',
        template: 'You are a helpful assistant. {context}',
        engine: 'f-string',
      },
    },
  },
  {
    resourcesId: 'rs-1234567890ab',
    version: 1,
    createTime: now - 30 * 24 * 3600,
    updateTime: now - 15 * 24 * 3600,
    resources: {
      legacy_llm: {
        resource_type: 'llm',
        endpoint: 'https://api.openai.com/v1',
        model: 'gpt-3.5-turbo-0301',
        sampling_parameters: {
          temperature: 0.9,
        },
      },
    },
  },
  {
    resourcesId: 'rs-abcdef123456',
    version: 5,
    createTime: now - 60 * 24 * 3600,
    updateTime: now - 12 * 3600,
    resources: {
      eval_llm: {
        resource_type: 'llm',
        endpoint: 'https://api.anthropic.com/v1',
        model: 'claude-3-sonnet-20240229',
        sampling_parameters: {
          temperature: 0.3,
          max_tokens: 2048,
        },
      },
      judge_template: {
        resource_type: 'prompt_template',
        template: 'Evaluate the following response: {response}\n\nCriteria: {criteria}',
        engine: 'jinja2',
      },
      dataset: {
        resource_type: 'dataset',
        path: 's3://my-bucket/eval-data/v1/',
        format: 'jsonl',
      },
    },
  },
  {
    resourcesId: 'rs-999888777666',
    version: 1,
    createTime: now - 2 * 24 * 3600,
    updateTime: now - 2 * 24 * 3600,
    resources: {
      test_llm: {
        resource_type: 'llm',
        endpoint: 'http://localhost:8080/v1',
        model: 'local-model',
        sampling_parameters: {
          temperature: 1.0,
        },
      },
    },
  },
];

const defaultHandlers = createResourcesHandlers(sampleResources);

function createStoryStore(configOverrides?: Partial<typeof initialConfigState>) {
  return createAppStore({
    config: {
      ...initialConfigState,
      baseUrl: STORY_BASE_URL,
      autoRefreshMs: 0,
      ...configOverrides,
    },
    rollouts: initialRolloutsUiState,
    resources: initialResourcesUiState,
  });
}

function renderWithStore(configOverrides?: Partial<typeof initialConfigState>) {
  const store = createStoryStore(configOverrides);

  return (
    <Provider store={store}>
      <>
        <ResourcesPage />
        <AppAlertBanner />
        <AppDrawerContainer />
      </>
    </Provider>
  );
}

function renderWithAppLayout(configOverrides?: Partial<typeof initialConfigState>) {
  const store = createStoryStore(configOverrides);
  const router = createMemoryRouter(
    [
      {
        path: '/',
        element: (
          <AppLayout
            config={{
              baseUrl: store.getState().config.baseUrl,
              autoRefreshMs: store.getState().config.autoRefreshMs,
            }}
          />
        ),
        children: [
          {
            path: '/resources',
            element: <ResourcesPage />,
          },
        ],
      },
    ],
    { initialEntries: ['/resources'] },
  );

  return (
    <Provider store={store}>
      <>
        <RouterProvider router={router} />
        <AppDrawerContainer />
      </>
    </Provider>
  );
}

export const Default: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
};

export const WithSidebarLayout: Story = {
  name: 'Within AppLayout',
  render: () => renderWithAppLayout(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
};

export const Search: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: defaultHandlers,
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await canvas.findByText('rs-a1b2c3d4e5f6');

    const searchInput = canvas.getByPlaceholderText('Search by Resources ID');
    await userEvent.type(searchInput, 'rs-abcdef123456');

    await waitFor(() => {
      if (canvas.queryByText('rs-a1b2c3d4e5f6')) {
        throw new Error('Expected search to filter out non-matching resources');
      }
      if (!canvas.queryByText('rs-abcdef123456')) {
        throw new Error('Expected matching resource to remain visible');
      }
    });
  },
};

export const EmptyState: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [http.get('*/v1/agl/resources', () => HttpResponse.json({ items: [], limit: 0, offset: 0, total: 0 }))],
    },
  },
};

export const ServerError: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/v1/agl/resources', () => HttpResponse.json({ detail: 'Internal server error' }, { status: 500 })),
      ],
    },
  },
};

export const RequestTimeout: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/v1/agl/resources', async () => {
          await delay(1200);
          return HttpResponse.json({ detail: 'Request timed out' }, { status: 504, statusText: 'Timeout' });
        }),
      ],
    },
  },
};

export const SingleResource: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createResourcesHandlers([sampleResources[0]]),
    },
  },
};

export const ParseFailure: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: [
        http.get('*/v1/agl/resources', () =>
          HttpResponse.text('{ malformed json', {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
            },
          }),
        ),
      ],
    },
  },
};

export const ManyResources: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createResourcesHandlers(
        Array.from({ length: 50 }, (_, index) => ({
          resourcesId: `rs-generated-${index + 1}`.padEnd(17, '0'),
          version: (index % 5) + 1,
          createTime: now - (50 - index) * 24 * 3600,
          updateTime: now - (50 - index) * 3600,
          resources: {
            llm: {
              resource_type: 'llm',
              endpoint: `https://api.example.com/v${(index % 3) + 1}`,
              model: index % 2 === 0 ? 'gpt-4' : 'claude-3-opus',
              sampling_parameters: {
                temperature: 0.5 + (index % 5) * 0.1,
              },
            },
          },
        })),
      ),
    },
  },
};

export const ComplexResources: Story = {
  render: () => renderWithStore(),
  parameters: {
    msw: {
      handlers: createResourcesHandlers([
        {
          resourcesId: 'rs-complex-123456',
          version: 10,
          createTime: now - 90 * 24 * 3600,
          updateTime: now - 600,
          resources: {
            primary_llm: {
              resource_type: 'llm',
              endpoint: 'https://api.openai.com/v1',
              model: 'gpt-4-turbo-preview',
              sampling_parameters: {
                temperature: 0.7,
                top_p: 0.95,
                frequency_penalty: 0.1,
                presence_penalty: 0.1,
                max_tokens: 4096,
              },
            },
            fallback_llm: {
              resource_type: 'llm',
              endpoint: 'https://api.anthropic.com/v1',
              model: 'claude-3-opus-20240229',
              sampling_parameters: {
                temperature: 0.7,
                max_tokens: 4096,
                top_k: 40,
              },
            },
            embedding_model: {
              resource_type: 'embedding',
              endpoint: 'https://api.openai.com/v1',
              model: 'text-embedding-3-large',
              dimensions: 1536,
            },
            system_prompt: {
              resource_type: 'prompt_template',
              template: 'You are an AI assistant. Context: {context}\nUser: {user_input}\nAssistant:',
              engine: 'jinja2',
              variables: ['context', 'user_input'],
            },
            training_dataset: {
              resource_type: 'dataset',
              path: 's3://ml-datasets/training/v2/data.jsonl',
              format: 'jsonl',
              size_bytes: 1024000000,
              num_examples: 50000,
            },
            validation_dataset: {
              resource_type: 'dataset',
              path: 's3://ml-datasets/validation/v2/data.jsonl',
              format: 'jsonl',
              size_bytes: 102400000,
              num_examples: 5000,
            },
          },
        },
      ]),
    },
  },
};

export const DarkTheme: Story = {
  render: () => renderWithStore({ theme: 'dark' }),
  parameters: {
    theme: 'dark',
    msw: {
      handlers: defaultHandlers,
    },
  },
};
