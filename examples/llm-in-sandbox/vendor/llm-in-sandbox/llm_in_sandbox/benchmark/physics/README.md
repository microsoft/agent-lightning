# Physics

UGPhysics undergraduate physics problems (650 problems).

> **Note:** This task uses **LLM-as-a-Judge** for evaluation — the benchmark run outputs placeholder scores, and you need a second step (`judge.py`) with a strong LLM to get real accuracy.

**Setup:** No extra dependencies.

**Step 1: Run benchmark**
```bash
llm-in-sandbox benchmark --task physics
llm-in-sandbox benchmark --task physics --mode llm   # vanilla LLM
```

**Step 2: Run LLM-as-Judge**

Requires Qwen3-235B-A22B-Instruct-2507 as judge model (local or remote):
```bash
python judge.py \
    --input output/<timestamp>_physics_<model>_<mode>/trajectory.json \
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

**Dataset:** `daixuancheng/llm-in-sandbox-bench` (config: `physics`)
