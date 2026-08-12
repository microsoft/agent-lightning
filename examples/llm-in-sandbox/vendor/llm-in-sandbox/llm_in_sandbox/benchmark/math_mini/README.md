# Math

AIME 2025 competition math problems (30 problems x 16 repeats).

**Setup:** `pip install math-verify`

**Run:**
```bash
llm-in-sandbox benchmark --task math
llm-in-sandbox benchmark --task math --mode llm   # vanilla LLM
```

**Metric:** Accuracy via `math-verify`. Answer format: `\boxed{answer}`.

**Dataset:** `daixuancheng/llm-in-sandbox-bench` (config: `math`)
