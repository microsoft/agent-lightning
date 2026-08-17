# Chemistry

ChemBench4K multiple choice questions (450 problems).

**Setup:** No extra dependencies.

**Run:**
```bash
llm-in-sandbox benchmark --task chem
llm-in-sandbox benchmark --task chem --mode llm   # vanilla LLM
```

**Metric:** Accuracy on multiple choice (A/B/C/D). Supports `\boxed{A}`, `Answer: A`, etc.

**Dataset:** `daixuancheng/llm-in-sandbox-bench` (config: `chem`)
