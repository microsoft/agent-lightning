# Biomedical QA

MedXpertQA multiple choice questions (500 problems).

**Setup:** No extra dependencies.

**Run:**
```bash
llm-in-sandbox benchmark --task biomed
llm-in-sandbox benchmark --task biomed --mode llm   # vanilla LLM
```

**Metric:** Accuracy on multiple choice (A-Z). Supports `\boxed{A}`, `Answer: A`, etc.

**Dataset:** `daixuancheng/llm-in-sandbox-bench` (config: `biomed`)
