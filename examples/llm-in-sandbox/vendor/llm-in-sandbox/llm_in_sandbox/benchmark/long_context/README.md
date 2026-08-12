# Long Context

AA-LCR long document reasoning (100 problems x 4 repeats).

> **Note:** This task uses **LLM-as-a-Judge** for evaluation — the benchmark run outputs placeholder scores, and you need a second step (`judge.py`) with a strong LLM to get real accuracy.

**Setup:** No extra dependencies.

**Step 1: Run benchmark**
```bash
# Use --max_token_limit 131072 to accommodate longer context
llm-in-sandbox benchmark --task long_context --max_token_limit 131072
llm-in-sandbox benchmark --task long_context --max_token_limit 131072 --mode llm   # vanilla LLM
```

**Step 2: Run LLM-as-Judge**

Requires Qwen3-235B-A22B-Instruct-2507 as judge model (local or remote):
```bash
python judge.py \
    --input output/<timestamp>_long_context_<model>_<mode>/trajectory.json \
    --judge_model qwen3-235B-A22B-instruct \
    --judge_base_url http://localhost:9991/v1 \
    --num_workers 32 \
    --api_key your_api_key
```

<details>
<summary>How to host the judge model locally</summary>

```bash
SAFETENSORS_FAST_GPU=1 vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --enable_expert_parallel --served-model-name qwen3-235B-A22B-instruct \
    --enable-prefix-caching \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 --port 9991 \
    --api_key your_api_key
```
</details>

**Metric:** LLM-as-a-Judge accuracy.

**Notes:**
- LLM-in-Sandbox mode: documents are provided as input files in the container
- Vanilla LLM mode: documents are provided directly in the prompt

**Dataset:** `daixuancheng/llm-in-sandbox-bench` (config: `long_context`)
