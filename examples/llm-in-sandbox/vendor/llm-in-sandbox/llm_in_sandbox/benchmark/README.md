# Benchmark

Reproduce our paper results, evaluate any LLM, or add your own tasks.

## Setup

### 1. Serve a Model (Skip this if you use API-based LLMs)

<details open>
<summary>Qwen3-Coder-30B-A3B-Instruct (vLLM)</summary>

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3_coder \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --tensor-parallel-size 8 \
    --enable-prefix-caching
```
</details>

<details>
<summary>Qwen3-4B-Instruct-2507 (vLLM)</summary>

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --served-model-name qwen3-4b-instruct \
    --enable-prefix-caching \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```
</details>

<details>
<summary>💡 Qwen3-4B-Instruct-2507 after LLM-in-Sandbox-RL (vLLM)</summary>

[daixuancheng/Qwen3-4B-Instruct-2507-LLM-in-Sandbox-RL](https://huggingface.co/daixuancheng/Qwen3-4B-Instruct-2507-LLM-in-Sandbox-RL) is the model checkpoint trained with [LLM-in-Sandbox-RL](https://github.com/llm-in-sandbox/llm-in-sandbox-rl) in our paper.
```bash
vllm serve daixuancheng/Qwen3-4B-Instruct-2507-LLM-in-Sandbox-RL \
    --served-model-name qwen3-4b-instruct-sandbox-rl \
    --enable-prefix-caching \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```
</details>


<details>
<summary>DeepSeek-V3.2-Thinking (SGLang)</summary>

```bash
python3 -m sglang.launch_server \
    --model-path "deepseek-ai/DeepSeek-V3.2" \
    --served-model-name "DeepSeek-V3.2" \
    --trust-remote-code \
    --tp-size 8 \
    --tool-call-parser deepseekv32 \
    --reasoning-parser deepseek-v3 \
    --host 0.0.0.0 \
    --port 8000
```
</details>

### 2. Once the server is ready, open a new terminal and set Environment Variables
<details open>
<summary>Environment Variables for Qwen3-Coder-30B-A3B-Instruct</summary>

```bash
export LLM_NAME=qwen3_coder                   # must match --served-model-name
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_NUM_WORKERS=8
# Temperature: 0.7 for Qwen3 series, 1.0 for all other LLMs evaluated in our paper
export LLM_TEMPERATURE=0.7
# export LLM_API_KEY=your-api-key             # optional for local servers
```

</details>

<details>
<summary>Environment Variables for Qwen3-4B-Instruct-2507</summary>

```bash
export LLM_NAME=qwen3-4b-instruct
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_NUM_WORKERS=8
# Temperature: 0.7 for Qwen3 series, 1.0 for all other LLMs evaluated in our paper
export LLM_TEMPERATURE=0.7
# export LLM_API_KEY=your-api-key             # optional for local servers
```

</details>

<details>
<summary>💡 Environment Variables for Qwen3-4B-Instruct-2507 after LLM-in-Sandbox-RL</summary>

```bash
export LLM_NAME=qwen3-4b-instruct-sandbox-rl
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_NUM_WORKERS=8
# Temperature: 0.7 for Qwen3 series, 1.0 for all other LLMs evaluated in our paper
export LLM_TEMPERATURE=0.7
# export LLM_API_KEY=your-api-key             # optional for local servers
```

</details>

<details>
<summary>Environment Variables for DeepSeek-V3.2-Thinking</summary>

```bash
export LLM_NAME=DeepSeek-V3.2
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_NUM_WORKERS=8
# Temperature: 0.7 for Qwen3 series, 1.0 for all other LLMs evaluated in our paper
export LLM_TEMPERATURE=1.0
# export LLM_API_KEY=your-api-key             # optional for local servers
```

> **Very Important!** DeepSeek-V3.2-Thinking requires an `--extra_body` flag when running benchmarks:
> ```bash
> llm-in-sandbox benchmark --task math ... --extra_body '{"chat_template_kwargs": {"thinking": True}}'
> ```
</details>

### 3. Run a Task

> Run the benchmark commands **in the same terminal session** where you exported the environment variables.

See each task folder for dependencies and details:

| Task | Description |
|------|-------------|
| [`math`](math/) | AIME 2025 competition problems |
| [`physics`](physics/) | Undergraduate physics |
| [`chem`](chem/) | Chemistry MCQ |
| [`biomed`](biomed/) | Biomedical QA |
| [`long_context`](long_context/) | Long document reasoning |
| [`instruct_follow`](instruct_follow/) | Instruction following |

**Example (math):**
```bash
# Install task-specific dependency
pip install math-verify
```
```bash
# LLM-in-Sandbox mode (default)
llm-in-sandbox benchmark --task math

# Vanilla LLM mode (for comparison)
llm-in-sandbox benchmark --task math --mode llm
```

### Parameters

All parameters can be set via CLI flags or environment variables. CLI flags take precedence.

| Parameter | Env Var | Description | Default |
|-----------|---------|-------------|---------|
| `--task` | | Task name (see table above) | *required* |
| `--llm_name` | `LLM_NAME` | Model name | *required* |
| `--llm_base_url` | `LLM_BASE_URL` | API endpoint URL | |
| `--api_key` | `LLM_API_KEY` | API key | |
| `--mode` | | `llm-in-sandbox` or `llm` (vanilla LLM) | `llm-in-sandbox` |
| `--num_workers` | `LLM_NUM_WORKERS` | Parallel workers | `1` |
| `--temperature` | `LLM_TEMPERATURE` | Sampling temperature | `1.0` |
| `--max_steps` | | Max agent steps (LLM-in-Sandbox mode) | `100` |
| `--max_response_len` | | Max response length (vanilla LLM mode only) | `65536` |
| `--extra_body` | | Extra JSON body for LLM API | |

Run `llm-in-sandbox benchmark --help` for all available parameters.

## Add Your Own Task

See [`demo/`](demo/) for a complete template.


## Troubleshooting

### Clean up Docker containers

If you interrupt the benchmark (Ctrl+C), some Docker containers may remain running. To clean them up:

```bash
# Clean up all containers from the default image
docker ps -aq --filter 'ancestor=cdx123/llm-in-sandbox:v0.1' | xargs -r docker rm -f

# Clean llm-in-sandbox related process
pkill -9 -f "llm-in-sandbox benchmark"
```

### Restart Docker (if containerd crashes)

If you see errors like `connection refused: containerd.sock`, Docker's backend has crashed. Restart it:

```bash
# Full restart (kill old processes, start containerd first, then dockerd)
pkill -9 dockerd 2>/dev/null; pkill -9 containerd 2>/dev/null
sleep 1 && nohup containerd > /var/log/containerd.log 2>&1 &
sleep 3 && rm -f /var/run/docker.pid && nohup dockerd > /var/log/dockerd.log 2>&1 &

# Check logs if something goes wrong
tail -f /var/log/containerd.log  # containerd logs
tail -f /var/log/dockerd.log     # dockerd logs
```

> Next time you may decrease `LLM_NUM_WORKERS` to avoid such errors.
