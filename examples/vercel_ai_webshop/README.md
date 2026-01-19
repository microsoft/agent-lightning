# WebShop Example

This example demonstrates how to train a Vercel AI SDK agent on the WebShop benchmark using Agent Lightning with reinforcement learning (VERL/GRPO). The training pipeline uses a headless TypeScript runner that executes agent rollouts and reports traces to the Agent Lightning coordinator.

## Requirements

- Node.js 22+ and pnpm 10+
- Docker (recommended) OR Python 3.8+ with Java 17+
- GPU with 40GB+ VRAM (for VERL training)
- HuggingFace token (`HF_TOKEN`)

## Quick Start

The recommended way to run the training pipeline is with Docker, which starts all services with a single command.

```bash
cd examples/vercel_ai_webshop

# 1. Set up environment
make setup

# 2. Run GPU training (VERL manages the Qwen model via vLLM)
make train
```

This starts:
- **WebShop Server** (`:3000`) - Flask shopping environment
- **Training Coordinator** (`:4747`) - Agent Lightning Store + VERL
- **Headless Runners** - Poll for tasks and execute agent rollouts

> **Note:** The first run downloads ~100MB of dataset files. This takes about 2 minutes but only happens once.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace token for model access | Yes |
| `WANDB_API_KEY` | Weights & Biases API key for metrics | No |
| `WEBSHOP_URL` | WebShop server URL | No (default: `http://localhost:3000`) |

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `agl/run_training.py` | Training coordinator entry point |
| `agl/config.py` | VERL/GRPO configuration (model, epochs, batch sizes) |
| `agl/tasks.py` | Task loading utilities (JSON, Parquet) |
| `agl/generate_tasks.py` | Generate tasks from WebShop human instruction data |
| `scripts/headless-runner.ts` | Headless rollout runner for training |
| `scripts/run_stack.sh` | Stack orchestration script |
| `src/agent/webshop-agent.ts` | ToolLoopAgent with Vercel AI SDK |
| `src/environment/webshop-server.ts` | HTTP client for WebShop Flask server |
| `src/utils/agentlightning/` | Store client, OpenTelemetry tracing, ProxyLLM utilities |
| `server/` | Python WebShop backend |
| `aml/` | Azure ML configuration files |

## Running Examples

### Training (Docker)

```bash
# Start GPU training - VERL manages vLLM, no API key needed
make train

# Run with more runners
N_RUNNERS=3 make train

# Check container status
make status

# Stop all services
make stop
```

### Training (Manual)

If you prefer to run services manually without Docker:

**Terminal 1 - WebShop Server:**
```bash
cd examples/vercel_ai_webshop
docker compose up webshop --build
```

**Terminal 2 - Training Coordinator:**
```bash
cd examples/vercel_ai_webshop/agl
./setup.sh                    # First time only
source activate.sh
python run_training.py qwen   # Full training
```

**Terminal 3+ - Headless Runners:**
```bash
cd examples/vercel_ai_webshop
export AGENT_LIGHTNING_STORE_URL="http://localhost:4747"
pnpm headless -- --worker-id runner-1
```

### Generating Tasks

By default, training uses `sample_tasks.json` with 8 tasks. For full training, generate tasks from the WebShop dataset:

```bash
# Generate all tasks (~12,000 tasks)
python agl/generate_tasks.py

# With custom options
python agl/generate_tasks.py --output agl/webshop_tasks.json --max-tasks 1000 --shuffle

# Train with generated tasks
python agl/run_training.py qwen --tasks-file agl/webshop_tasks.json
```

## Running on Azure ML

The `aml/` directory contains Azure ML configuration for running training jobs in the cloud. The job runs all services in a single container on a GPU node.

### Prerequisites

1. Install Azure CLI with ML extension:
   ```bash
   az extension add -n ml
   az login
   ```

2. Set environment variables:
   ```bash
   export AZURE_SUBSCRIPTION_ID="your-subscription-id"
   export HF_TOKEN="your-huggingface-token"
   export WANDB_API_KEY="your-wandb-api-key"
   ```

### Submit Job

```bash
# One-time setup (creates compute cluster)
make aml-setup

# Submit training job
make aml-train

# Stream logs
make aml-logs

# Check job status
make aml-status
```

### Using az ml CLI directly

```bash
RG=<your-resource-group>
WS=<your-workspace>

# Create compute cluster (one-time)
az ml compute create -f aml/compute.yml -g $RG -w $WS

# Submit job
az ml job create -f aml/jobs/webshop-qwen.yml --stream \
  --set environment_variables.HF_TOKEN="$HF_TOKEN" \
  --set environment_variables.WANDB_API_KEY="$WANDB_API_KEY" \
  -g $RG -w $WS

# Stream logs
az ml job stream -n <job-name> -g $RG -w $WS
```

### Customization

```bash
# Change number of runners
az ml job create -f aml/jobs/webshop-qwen.yml --stream \
  --set environment_variables.N_RUNNERS=4 \
  --set environment_variables.HF_TOKEN="$HF_TOKEN" \
  -g $RG -w $WS

# Use different compute
az ml compute create --name my-gpu-cluster --size Standard_NC48ads_A100_v4 \
  --min-instances 0 --max-instances 2 -g $RG -w $WS
az ml job create -f aml/jobs/webshop-qwen.yml --set compute=azureml:my-gpu-cluster ...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `connect ECONNREFUSED` | Wait for service healthcheck or run `make status` |
| Container `Exited (1)` | Check logs: `docker compose logs` |
| Port 3000 in use | Set `WEBSHOP_URL=http://localhost:3001` in `.env` |
| WebShop data download fails | Check network access; data downloads from Google Drive |
| AML compute not starting | Check quota limits and VM availability in your region |
| vLLM/flash-attn build errors | Ensure `VLLM_USE_V1=1` is set; check CUDA 12.6+ support |

## Related

- [AI SDK Documentation](https://sdk.vercel.ai/docs)
- [WebShop Benchmark](https://github.com/princeton-nlp/WebShop)
- [Agent Lightning Docs](../../docs/)
