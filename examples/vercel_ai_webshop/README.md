# WebShop Training Pipeline with Agent Lightning

This example demonstrates how to train a Vercel AI SDK agent on the WebShop benchmark using Agent Lightning (agl). The training pipeline uses a **headless runner** that executes agent rollouts, reports traces to the Agent Lightning coordinator, and supports both CPU-based prototyping (with external LLMs) and GPU-based RL training (with VERL/vLLM).

## Overview

The training pipeline consists of:

1. **Training Coordinator (agl)** — Manages the task queue, collects traces, and runs the training algorithm
2. **Headless Runner** — Executes the Vercel AI agent in a loop, polling for tasks and reporting rewards
3. **WebShop Server** — The shopping environment (Flask + OpenAI Gym)

## Features

- **Headless Execution**: Run agent rollouts without a browser using the TypeScript headless runner
- **Agent Lightning Integration**: Full tracing via OpenTelemetry, task queue management, and reward reporting
- **VERL/vLLM Support**: GPU training mode with automatic LLM endpoint discovery
- **Scalable Runners**: Launch multiple headless runners in parallel for faster rollout collection
- **Dev Mode**: CPU-only prototyping using external LLM endpoints (OpenAI, local vLLM, etc.)

## Prerequisites

- Node.js 22+
- pnpm 10+
- Docker (recommended) OR Python 3.8+ with Java 17+ (for the WebShop server)
- OpenAI API key (only needed for dev mode with external LLM)

## Quick Start

The recommended way to run the training pipeline is with Docker Compose, which starts all services with a single command.

```bash
cd examples/vercel_ai_webshop

# 1. Set up environment
make setup

# 2. Run GPU training (VERL manages the Qwen model via vLLM)
make train

# Or run dev mode for CPU-only prototyping with an external LLM
# Edit .env and set OPENAI_API_KEY + OPENAI_API_BASE first
make dev
```

This starts:
- **WebShop**: http://localhost:3000 — Shopping environment API
- **Store**: http://localhost:4747 — Agent Lightning coordinator
- **Headless Runner** — Polls for tasks and executes agent rollouts

In **GPU training mode** (`make train`), VERL manages vLLM internally with the Qwen model—no external API key is needed. The headless runners automatically discover the LLM endpoint from the Store.

In **dev mode** (`make dev`), runners connect to an external LLM endpoint configured via `OPENAI_API_BASE`. This is useful for CPU-only prototyping or testing with different models.

> **Note:** The first run downloads ~100MB of dataset files. This takes about 2 minutes but only happens once (data is persisted in a Docker volume).

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | API key for external LLM endpoint | Dev mode only |
| `OPENAI_API_BASE` | OpenAI-compatible endpoint URL | Dev mode only |
| `WEBSHOP_URL` | WebShop server URL | No (default: `http://localhost:3000`) |
| `WANDB_API_KEY` | Weights & Biases API key for metrics tracking | No (recommended) |

Training metrics (rewards, success rates, response lengths) are automatically logged to [Weights & Biases](https://wandb.ai). Set `WANDB_API_KEY` to enable experiment tracking.

## How It Works

### Training Architecture

The training pipeline consists of three main components that communicate via REST APIs and OpenTelemetry:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Training Coordinator (agl)                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │ Task Queue      │    │ Algorithm       │    │ Trace Collector │      │
│  │ (enqueue tasks) │    │ (VERL/Baseline) │    │ (OTLP receiver) │      │
│  └────────┬────────┘    └─────────────────┘    └────────▲────────┘      │
└───────────┼──────────────────────────────────────────────┼──────────────┘
            │ REST API                              OTLP HTTP
            │ (dequeue/update)                    (/v1/traces)
            ▼                                              │
┌───────────────────────┐                                  │
│    Headless Runner    │──────────────────────────────────┘
│   (TypeScript/Node)   │
│  ┌─────────────────┐  │     ┌─────────────────┐
│  │ WebShop Agent   │◄─┼────►│ LLM Endpoint    │
│  │ (Vercel AI SDK) │  │     │ (OpenAI/vLLM)   │
│  └────────┬────────┘  │     └─────────────────┘
└───────────┼───────────┘
            │ HTTP (search, click, buy)
            ▼
┌───────────────────────┐
│    WebShop Server     │
│   (Flask @ :3000)     │
└───────────────────────┘
```

### Key Components

1. **Training Coordinator (`agl/run_training.py`)**: Manages the task queue, collects traces via OTLP, and runs the training algorithm (Baseline for dev mode, VERL for GPU training)
2. **Headless Runner (`scripts/headless-runner.ts`)**: Polls for tasks, executes the Vercel AI agent, and reports rewards back to the coordinator
3. **WebShop Agent (`src/agent/webshop-agent.ts`)**: Uses `ToolLoopAgent` with three tools (search, click, buy) to navigate the store
4. **WebShopServerEnv**: HTTP adapter that connects to the WebShop Python server
5. **Store Client (`src/utils/agentlightning/`)**: REST client for task queue operations and LLM endpoint discovery

### Action Grammar

The agent interacts with the WebShop server using a simple action grammar:

- `search[query]` - Search for products matching the query
- `click[element]` - Click on an element (product, option, button)
- `click[Buy Now]` - Complete the purchase (convenience: `buy` tool)

### Sample Tasks

| Task ID | Description |
|---------|-------------|
| ws_001 | Red cotton t-shirt, men, size L, under $30 |
| ws_002 | Black running shorts, men, M, athletic style |
| ws_003 | Gray fleece hoodie, women, size S |
| ws_004 | White canvas sneakers, size 9, under $50 |
| ws_005 | Navy slim fit chino pants, waist 32, length 32 |
| ws_006 | Black yoga leggings, women, size M |
| ws_007 | White organic cotton blouse, women, size M |
| ws_008 | Navy polo shirt, men, size XL, under $50 |


## Running the Training Pipeline

The training pipeline uses Docker Compose to orchestrate all services. This section covers configuration options, monitoring, and manual setup alternatives.

### Architecture

The training pipeline uses a **unified container** that runs all three services (WebShop, Coordinator, Runners) as processes within a single container. This architecture:

- **Simplifies deployment** - Single image to build and deploy
- **Enables localhost communication** - All services communicate via `127.0.0.1`
- **Matches Azure ML pattern** - Same architecture works locally and in the cloud
- **Isolates dependencies** - Python 3.11 + Java 21 + Node.js 20 in one image

| Component | Runtime | Port | Description |
|-----------|---------|------|-------------|
| **WebShop Server** | Python + Java 21 | 3000 | Flask shopping environment with pyserini search |
| **Training Coordinator** | Python | 4747 | Agent Lightning Store + training algorithm |
| **Headless Runners** | Node.js 20 | - | Vercel AI SDK agent execution |

```
┌─────────────────────────────────────────────────────────────┐
│                 Single Container (webshop-agl)              │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  WebShop    │  │    AGL      │  │   N x Runners       │  │
│  │   Server    │  │ Coordinator │  │   (Node.js)         │  │
│  │  :3000      │  │   :4747     │  │                     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │             │
│         └────────────────┴────────────────────┘             │
│                    localhost                                │
└─────────────────────────────────────────────────────────────┘
```

### Inspecting Logs

Logs are automatically saved to `logs/<profile>-<timestamp>.log` when running `make dev` or `make train`. This makes it easy to inspect what happened after a training run without needing to keep a terminal open.

**Log files:**
```bash
make logs-latest  # Tail the most recent log file
ls logs/          # List all saved log files
```

**Live log streaming:**
```bash
make logs         # Follow all logs (live)
make status       # Show all container states
```

**Verify LLM endpoint configuration:**
```bash
docker compose logs 2>&1 | grep -E "LLM resource|OPENAI_API_BASE|Using"
```

Expected output in dev mode:
```
[runner-1] No LLM resource in Store, using OPENAI_API_BASE fallback
[runner-1] Using OPENAI_API_BASE: https://api.openai.com/v1
```

In GPU mode (with VERL), you'll see:
```
[runner-1] Found ProxyLLM resource: Qwen/Qwen2.5-1.5B-Instruct @ http://...
[runner-1] Using LLM Proxy: http://.../rollout/{id}/attempt/{id}/v1
```

**Verify tasks are being processed:**
```bash
docker compose logs 2>&1 | grep -E "PROGRESS|Enqueued|rollout"
```

**Filtered views for training:**
```bash
make watch-steps      # Agent decisions only (most useful)
make watch-progress   # Training coordinator status
make watch            # 3-pane tmux view (requires tmux)
```

**Troubleshooting common issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No LLM resource in Store` | Expected in dev mode | Ensure `OPENAI_API_BASE` is set in `.env` |
| `Incorrect API key` | Invalid `OPENAI_API_KEY` | Update `.env` with valid key |
| `connect ECONNREFUSED` | Service not ready | Wait for healthcheck or run `make status` |
| Container `Exited (1)` | Service crashed | Check logs: `make logs-<service>` |

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create `.env` from template |
| `make build` | Build Docker image |
| `make dev` | Start dev stack (single container, CPU) |
| `make train` | Start training stack (single container, GPU) |
| `make scale N=3` | Set number of runners |
| `make watch` | Launch tmux with training visibility |
| `make watch-steps` | Follow only agent step logs (compact) |
| `make watch-progress` | Follow only training progress logs |
| `make logs` | Follow all logs (live) |
| `make logs-latest` | Tail the most recent log file |
| `make status` | Show container status |
| `make stop` | Stop all services |
| `make clean` | Stop, remove volumes and images |
| `make aml-setup` | One-time Azure ML setup (compute + environment) |
| `make aml-train` | Submit training job to Azure ML |
| `make aml-logs` | Stream logs from running AML job |
| `make aml-status` | Show AML job status |

### Docker Features

- **Automatic Log Collection**: Logs are saved to `logs/<profile>-<timestamp>.log` for post-run inspection.
- **Data Persistence**: The WebShop dataset (~100MB) is stored in a Docker volume (`webshop_data`). It downloads once on first run and persists across rebuilds.
- **Unified Setup**: No local Node.js or Python installation required.

### Training Visibility

Use `make watch` to launch a tmux session with 3 panes showing training activity in order of importance:

```
+---------------------------------------------------------------+
| AGENT DECISIONS (top, largest pane)                           |
| [TASK] runner-1 | rollout=abc123... | Find a red t-shirt...  |
| [STEP] 1. search[red t-shirt] →                               |
| [STEP] 2. click[B07XYZ123] →                                  |
| [STEP] 3. click[Buy Now] ✓ reward=1.00                        |
| [DONE] runner-1 | reward=1.00 | steps=3 | SUCCESS             |
+---------------------------------------------------------------+
| TRAINING PROGRESS (middle pane)                               |
| [PROGRESS] Running 10 tasks (max_tasks=10)                    |
| [PROGRESS] External runners should connect to execute tasks   |
+---------------------------------------------------------------+
| WEBSHOP SERVER (bottom, smallest pane)                        |
| webshop | INFO - Created session sess_abc123                  |
+---------------------------------------------------------------+
```

- **Top pane**: Agent decisions - shows task starts, each action step, and completion status
- **Middle pane**: Training coordinator progress - shows task queue and training status
- **Bottom pane**: WebShop server logs - for debugging environment issues

Requires `tmux` to be installed. Use `Ctrl+B D` to detach from the session.

### Docker Compose Profiles

Two profiles are available, both using the unified container architecture:

- **`dev`** - CPU-only mode. Runs all services in a single container. Uses `OPENAI_API_BASE` for LLM inference (configure in `.env`).
- **`gpu`** - GPU training mode. Runs all services in a single container with GPU access. VERL manages vLLM internally.

```bash
# GPU mode (recommended) - VERL manages vLLM, no API key needed
docker compose --profile gpu up --build

# Dev mode - requires OPENAI_API_KEY and OPENAI_API_BASE in .env
docker compose --profile dev up --build

# Run with more runners
N_RUNNERS=3 docker compose --profile dev up --build
```

### Manual Setup (Without Docker)

If you prefer to run services manually without Docker, follow the instructions below. This is useful for development or debugging individual components.

**Terminal 1 - WebShop Server:**
```bash
cd examples/vercel_ai_webshop
docker compose up webshop --build
```

**Terminal 2 - Training Coordinator (Dev Mode):**
```bash
cd examples/vercel_ai_webshop/agl

# First time setup (creates venv and installs dependencies)
./setup.sh

# Activate the environment
source activate.sh

# Run dev mode with 5 tasks (default)
python run_training.py --dev

# Or specify more tasks
python run_training.py --dev --max-tasks 10
```

**Terminal 3 - Headless Runner:**
```bash
cd examples/vercel_ai_webshop

export AGENT_LIGHTNING_STORE_URL="http://localhost:4747"

# For dev mode only: set external LLM endpoint
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # or local vLLM, etc.

# Run the headless agent
pnpm headless -- --worker-id runner-1
```

The headless runner will:
1. Connect to the Agent Lightning Store
2. Check for LLM resources published by VERL (GPU mode)
3. If no resources found, fall back to `OPENAI_API_BASE` (dev mode)
4. Poll for tasks and execute the WebShop agent
5. Report traces and rewards back to the coordinator

---

### Full Training with VERL (GPU Required)

For RL-based fine-tuning with VERL, you need a GPU with 40GB+ memory.

**Terminal 1 - WebShop Server:**
```bash
cd examples/vercel_ai_webshop
docker compose up webshop --build
```

**Terminal 2 - Training Coordinator:**
```bash
cd examples/vercel_ai_webshop/agl

# First time setup with GPU support
./setup.sh --gpu

# Activate the environment
source activate.sh

# Fast training for CI/testing (smaller model)
python run_training.py fast

# Full Qwen training
python run_training.py qwen

# With custom tasks file
python run_training.py qwen --tasks-file data/tasks.json
```

**Terminal 3+ - Headless Runners (one or more):**
```bash
cd examples/vercel_ai_webshop
export AGENT_LIGHTNING_STORE_URL="http://localhost:4747"

# Run multiple workers for parallel rollouts
pnpm headless -- --worker-id runner-1
# In another terminal:
pnpm headless -- --worker-id runner-2
```

---

### vLLM Integration

When training with VERL, the system uses vLLM to serve the Qwen model locally instead of calling external APIs. This enables:

- **Automatic model loading**: VERL starts vLLM with the configured model (e.g., `Qwen/Qwen2.5-1.5B-Instruct`)
- **LLM Proxy**: An OpenAI-compatible proxy wraps vLLM to capture token IDs and emit traces
- **Dynamic endpoint discovery**: Runners fetch the LLM endpoint from the Store (no manual configuration)
- **Trace attribution**: Each request is routed with rollout/attempt IDs for accurate training

**Architecture with vLLM:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          VERL Training Coordinator                          │
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────────┐ │
│  │ Task Queue   │   │ vLLM Server  │   │ LLM Proxy                        │ │
│  │              │   │ (Qwen model) │◄──┤ - OpenAI-compatible API          │ │
│  │              │   │              │   │ - Token ID capture               │ │
│  │              │   │              │   │ - Rollout/attempt routing        │ │
│  └──────┬───────┘   └──────────────┘   └──────────────┬───────────────────┘ │
└─────────┼──────────────────────────────────────────────┼────────────────────┘
          │ GET /v1/agl/resources/latest                 │
          │ POST /v1/agl/queues/rollouts/dequeue         │
          ▼                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Headless Runner                                    │
│  1. Fetch resources from Store → discovers ProxyLLM endpoint                 │
│  2. For each rollout, construct routed URL:                                   │
│     {proxy_endpoint}/rollout/{rollout_id}/attempt/{attempt_id}/v1            │
│  3. Use routed URL with Vercel AI SDK's createOpenAI()                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Endpoint Discovery:**

The headless runner automatically discovers the LLM endpoint:

1. On startup, queries `GET /v1/agl/resources/latest` from the Store
2. Extracts the `main_llm` resource (a `ProxyLLM` with the vLLM endpoint)
3. For each rollout, constructs a routed URL with rollout/attempt IDs
4. Falls back to `OPENAI_API_BASE` environment variable if no resource found

**Configuration in `agl/config.py`:**

```python
RL_TRAINING_CONFIG = {
    "actor_rollout_ref": {
        "rollout": {
            "name": "vllm",  # Use vLLM for inference
            "gpu_memory_utilization": 0.8,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "model": {
            "path": "Qwen/Qwen2.5-1.5B-Instruct",  # HuggingFace model ID
        },
    },
    # ...
}
```

**Dev Mode vs Training Mode:**

| Mode | Model Serving | How Endpoint is Configured |
|------|---------------|---------------------------|
| Dev (`--dev`) | External API | `OPENAI_API_BASE` env var |
| Training (`fast`, `qwen`) | VERL-managed vLLM | Auto-discovered from Store |

---

### Headless Runner Reference

The headless runner (`scripts/headless-runner.ts`) executes agent rollouts outside of a browser environment.

**Usage:**
```bash
pnpm headless -- [options]
# Or directly:
npx tsx scripts/headless-runner.ts [options]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--worker-id <id>` | Unique worker identifier | `runner-{timestamp}` |
| `--poll-interval <ms>` | Task polling interval | `1000` |
| `--max-steps <n>` | Maximum steps per task | `15` |
| `--once` | Run single task and exit | `false` |

**Environment Variables:**

| Variable | Description | Required |
|----------|-------------|----------|
| `AGENT_LIGHTNING_STORE_URL` | Store server URL (e.g., `http://localhost:4747`) | Yes |
| `OPENAI_API_KEY` | API key for external LLM endpoint | Dev mode only |
| `OPENAI_API_BASE` | OpenAI-compatible endpoint URL | Dev mode only |
| `WEBSHOP_MODEL` | Model ID for inference | No (default: `gpt-4o-mini`) |
| `WEBSHOP_URL` | WebShop server URL | No (default: `http://localhost:3000`) |

In GPU mode with VERL, the runner automatically discovers the LLM endpoint from the Store—no API key or base URL needed.

---

### Training Coordinator Reference

The training coordinator (`agl/run_training.py`) manages the task queue and training loop.

**Usage:**
```bash
cd examples/vercel_ai_webshop/agl
python run_training.py [config] [options]
```

**Configurations:**

| Config | Model | GPU Required | Use Case |
|--------|-------|--------------|----------|
| `--dev` | External (via `OPENAI_API_BASE`) | No | CPU-only prototyping |
| `fast` | Qwen2.5-0.5B-Instruct | Yes (40GB) | CI testing |
| `qwen` | Qwen2.5-1.5B-Instruct | Yes (40GB) | Full training |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dev` | Run in dev mode (Baseline algorithm, no GPU) | - |
| `--max-tasks N` | Max tasks in dev mode | `5` |
| `--tasks-file PATH` | Custom tasks file (JSON/Parquet) | Sample tasks |
| `--val-tasks-file PATH` | Custom validation tasks | First 10 training tasks |

---

## Included Files

### Training Coordinator (`agl/`)

| File | Description |
|------|-------------|
| `run_training.py` | Main entry point for training with dev/VERL modes |
| `config.py` | VERL/GRPO configuration (model, epochs, batch sizes) |
| `tasks.py` | Task loading utilities (JSON, Parquet, sample tasks) |

### Source Code (`src/`)

| File | Description |
|------|-------------|
| `agent/webshop-agent.ts` | ToolLoopAgent for UI with Vercel AI SDK |
| `agent/prompts.ts` | System prompts shared between UI and headless runner |
| `environment/webshop-server.ts` | HTTP client for WebShop Flask server |

### Agent Lightning Integration (`src/utils/agentlightning/`)

| File | Description |
|------|-------------|
| `store-client.ts` | REST client for Store API (dequeue, complete, resources) |
| `otel.ts` | OpenTelemetry tracer factory with rollout attribution |
| `proxy-llm.ts` | ProxyLLM URL construction for rollout/attempt routing |
| `types.ts` | TypeScript types matching Python models |
| `index.ts` | Re-exports all utilities |

### Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `headless-runner.ts` | Headless rollout runner for training |
| `run_stack.sh` | Stack orchestration script |
| `watch-training.sh` | Training visibility (tmux) |

## Project Structure

```
vercel_ai_webshop/
├── src/                              # TypeScript source code
│   ├── agent/                        # Agent definition
│   │   ├── webshop-agent.ts          # ToolLoopAgent for Next.js UI
│   │   └── prompts.ts                # System prompts (shared)
│   ├── environment/                  # WebShop HTTP client
│   │   └── webshop-server.ts         # HTTP client for WebShop API
│   ├── data/                         # Task definitions
│   └── utils/agentlightning/         # Agent Lightning integration
│       ├── index.ts                  # Re-exports
│       ├── store-client.ts           # Store REST client
│       ├── otel.ts                   # OpenTelemetry tracing
│       ├── proxy-llm.ts              # ProxyLLM URL helpers
│       └── types.ts                  # TypeScript types
├── scripts/
│   ├── headless-runner.ts            # Headless rollout runner
│   ├── run_stack.sh                  # Stack orchestration script
│   └── watch-training.sh             # Training visibility (tmux)
├── agl/                              # Python Agent Lightning code
│   ├── run_training.py               # Training coordinator
│   ├── config.py                     # Training configurations
│   └── tasks.py                      # Task loading utilities
├── server/                           # Python WebShop backend
│   └── webshop_server.py             # Flask server
├── aml/                              # Azure ML configuration
│   ├── compute.yml                   # GPU compute cluster
│   ├── environment.yml               # Training environment
│   ├── job.yml                       # Training job spec
│   └── README.md                     # AML instructions
├── docker-compose.yml                # Service orchestration
├── Dockerfile                        # Unified training image
└── Makefile                          # Build and run commands
```

## Agent Lightning Integration

The `src/utils/agentlightning/` directory provides TypeScript utilities that enable training Vercel AI SDK agents with Agent Lightning. These utilities handle task queue management, OpenTelemetry tracing, and LLM endpoint discovery.

### Store Client (`store-client.ts`)

REST client for the Agent Lightning Store server:

- **Task Queue**: `dequeueRollouts()` / `completeAttempt()` - poll for work and report results
- **Resources**: `getLatestResources()` - discover vLLM endpoint dynamically
- **Health**: `health()` - check server availability

```typescript
import { AgentLightningStoreClient } from './utils/agentlightning';

const client = new AgentLightningStoreClient({ baseUrl: 'http://localhost:4747' });
const rollouts = await client.dequeueRollouts(1, 'runner-1');
// ... execute rollout ...
await client.completeAttempt(rolloutId, attemptId, { success: true, reward: 1.0 });
```

### OpenTelemetry Tracing (`otel.ts`)

Creates per-rollout tracers that emit spans to the Agent Lightning Store:

- `createRolloutTracer()` - factory that embeds rollout_id/attempt_id in Resource attributes
- `emitLlmCallSpan()` - emit LLM call with tokenized prompt/response for training
- `emitReward()` - emit reward span that the daemon extracts for training signal
- `getOtlpEndpoint()` - derive OTLP endpoint from Store URL

```typescript
import { createRolloutTracer, emitLlmCallSpan, emitReward } from './utils/agentlightning';

const { tracer, provider } = createRolloutTracer({
  otlpEndpoint: 'http://localhost:4747/v1/traces',
  serviceName: 'webshop-runner',
  rolloutId: 'ro-abc123',
  attemptId: 'at-def456',
});

// During agent execution, emit LLM calls with token IDs for training
emitLlmCallSpan(tracer, promptText, responseText, modelId);

// At the end, emit the final reward
emitReward(tracer, reward);

// Flush traces before completing the attempt
await provider.forceFlush();
```

### ProxyLLM Routing (`proxy-llm.ts`)

Utilities for constructing routed LLM endpoints that enable trace attribution:

- `getProxyLLMBaseUrl()` - construct `{endpoint}/rollout/{id}/attempt/{id}/v1`
- `getMainLLM()` - extract main_llm resource from Store response
- `isProxyLLM()` - type guard for ProxyLLM vs regular LLM

```typescript
import { getMainLLM, getProxyLLMBaseUrl, isProxyLLM } from './utils/agentlightning';

const resources = await client.getLatestResources();
const llmResource = getMainLLM(resources);

if (llmResource && isProxyLLM(llmResource)) {
  // Construct routed URL for proper trace attribution
  const baseURL = getProxyLLMBaseUrl(llmResource, rolloutId, attemptId);
  // baseURL = "http://proxy:8080/rollout/ro-abc123/attempt/at-def456/v1"
}
```

### Integration Flow

The following diagram shows how the TypeScript utilities integrate with the training loop:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Training Loop                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DEQUEUE TASK                                                             │
│     └─► storeClient.dequeueRollouts()                                       │
│         Returns: { rollout_id, attempt_id, input: { instruction, ... } }    │
│                                                                              │
│  2. DISCOVER LLM ENDPOINT                                                    │
│     └─► storeClient.getLatestResources()                                    │
│     └─► getMainLLM(resources) → ProxyLLM or LLM resource                   │
│     └─► getProxyLLMBaseUrl(resource, rolloutId, attemptId)                 │
│                                                                              │
│  3. CREATE TRACER                                                            │
│     └─► createRolloutTracer({ otlpEndpoint, rolloutId, attemptId, ... })   │
│         Returns: { tracer, provider }                                        │
│                                                                              │
│  4. EXECUTE AGENT LOOP                                                       │
│     for each step:                                                           │
│       └─► LLM call via Vercel AI SDK                                        │
│       └─► emitLlmCallSpan(tracer, prompt, response, model)                  │
│       └─► Execute action in WebShop environment                             │
│                                                                              │
│  5. EMIT REWARD                                                              │
│     └─► emitReward(tracer, finalReward)                                     │
│                                                                              │
│  6. FLUSH & COMPLETE                                                         │
│     └─► provider.forceFlush()  // Send all spans to Store                   │
│     └─► storeClient.completeAttempt(rolloutId, attemptId, { reward })       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The training coordinator (Python) collects spans via the OTLP endpoint, extracts token IDs from `emitLlmCallSpan()` spans, and uses the reward from `emitReward()` spans to compute policy gradients for GRPO training.

## Running on Azure ML

The `aml/` directory contains Azure ML configuration files for running training jobs in the cloud. See [aml/README.md](aml/README.md) for detailed instructions.

### Quick Start

```bash
# Prerequisites
az extension add -n ml
az login
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export HF_TOKEN="your-huggingface-token"
export WANDB_API_KEY="your-wandb-api-key"

# One-time setup
make aml-setup

# Submit training job
make aml-train

# Stream logs
make aml-logs
```

### How It Works

Azure ML jobs use the same unified container architecture as local development. The `scripts/run_stack.sh` script runs all three services (WebShop, Coordinator, Runners) as processes within a single container on a GPU node:

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure ML GPU Node                         │
│                  (Standard_NC24ads_A100_v4)                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Single Container (webshop-agl-gpu)       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │  │  WebShop    │  │    AGL      │  │   Runners   │   │   │
│  │  │   :3000     │  │   :4747     │  │  (N procs)  │   │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │   │
│  │         └────────────────┴────────────────┘          │   │
│  │                    localhost                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### WebShop Server Not Running

If you see connection errors, ensure the Python server is running:

```bash
# In the server directory
cd examples/vercel_ai_webshop/server
source activate.sh
python webshop_server.py
```

### Port Conflicts

If port 3000 is already in use:

```bash
# Run WebShop on a different port
python webshop_server.py --port 3001

# Update .env
WEBSHOP_URL=http://localhost:3001
```

### Python Version Issues

The WebShop environment requires Python 3.8+. Check your version:

```bash
python3 --version
```

If needed, install Python 3.8 or later from [python.org](https://www.python.org/downloads/) or using your system's package manager.

### Setup Script Fails

If `setup.sh` fails to download data, you can try manually:

```bash
cd server/webshop
# Follow WebShop's README for manual setup
```

## Related

- [AI SDK Documentation](https://sdk.vercel.ai/docs)
- [WebShop Benchmark](https://github.com/princeton-nlp/WebShop)
