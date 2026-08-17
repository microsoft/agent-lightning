# Demo Task (Template for Custom Tasks)

This is a **complete example** showing how to create a custom benchmark task.

## Files Structure

```
benchmark/demo/
├── config.yaml              # Dataset + prompt config (REQUIRED)
├── reward.py                # Scoring function (REQUIRED)
├── vanilla_llm_prompt.py    # Prompt for vanilla LLM mode (OPTIONAL)
├── data.json                # Local data (OPTIONAL, can use HuggingFace)
└── README.md                # Documentation (OPTIONAL)
```

## Required Files

### 1. `config.yaml`

Contains dataset source and prompt configuration for LLM-in-Sandbox mode.

### 2. `reward.py`

Must define `compute_score(agent_answer, ground_truth, **kwargs) -> float`.

## Optional Files

### 3. `vanilla_llm_prompt.py`

Custom prompt for vanilla LLM mode (`--mode llm`). If not provided, uses `problem_statement` directly.

### 4. `data.json`

Local dataset format (if not using HuggingFace):

## Run

```bash
# LLM-in-Sandbox
llm-in-sandbox benchmark --task demo

# Vanilla LLM
llm-in-sandbox benchmark --task demo --mode llm
```