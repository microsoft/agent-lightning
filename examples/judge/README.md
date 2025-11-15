# LLM-as-Judge Example

This example demonstrates how to evaluate and optimize AI agents using LLM-as-Judge with Agent Lightning's APO (Automatic Prompt Optimization). It's compatible with Agent-lightning v0.2 or later.

## Requirements

This example requires OpenAI's Python SDK. Install the required dependencies with:

```bash
pip install openai pydantic rich
```

Additionally, follow the [installation guide](../../docs/tutorials/installation.md) to install Agent-Lightning with APO support:

```bash
pip install "agent-lightning[apo]"
```

## Dataset

The example includes a sample dataset generator that creates QA pairs for demonstration purposes. You can also provide your own dataset in JSONL format.

### Dataset Format

Each line in `train.jsonl` / `val.jsonl` should follow this structure:

```json
{
  "id": "qa_001",
  "question": "What is the capital of France?",
  "reference_answer": "Paris is the capital and largest city of France.",
  "domain": "geography"
}
```

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `qa_agent.py` | QA agent implementation with LLM judge and APO integration |
| `qa_train.jsonl` | Training dataset (5 QA pairs, generated or custom) |
| `qa_val.jsonl` | Validation dataset (2 QA pairs, generated or custom) |

## Running Examples

### Generate Sample Data

First, generate sample QA pairs for testing:

```bash
python qa_agent.py --mode generate-data
```

This creates:
- `qa_train.jsonl` - 5 training examples
- `qa_val.jsonl` - 2 validation examples

### Baseline Evaluation

Test the baseline prompt without optimization. Set your OpenAI API key first:

```bash
export OPENAI_API_KEY=<your_api_key>
python qa_agent.py --mode baseline --limit 2
```

**Expected output**:
```
=== Baseline Evaluation ===
Task 1/2
=== Task ===
Question: What is the process of mitosis?
=== Prompt ===
Answer this question: What is the process of mitosis?
=== Answer ===
Mitosis is a type of cell division...
=== Judge Evaluation ===
Accuracy: 0.85
Completeness: 0.70
Clarity: 0.80
Reasoning: The answer is accurate and clear...
Overall: 0.78
=== Reward ===
0.78

Average reward: 0.75
Tasks evaluated: 2
```

### APO Training

Optimize the prompt using APO:

```bash
export OPENAI_API_KEY=<your_api_key>
python qa_agent.py --mode train --iterations 5
```

**Note**: Training requires multiple LLM calls and may take 5-10 minutes depending on the number of iterations.

### Evaluate Optimized Prompt

After training, evaluate with the optimized prompt:

```bash
export OPENAI_API_KEY=<your_api_key>
python qa_agent.py --mode eval --prompt outputs/best_prompt.txt
```

### Debug Mode

Inspect traces for a single task:

```bash
export OPENAI_API_KEY=<your_api_key>
python qa_agent.py --mode debug --task 0
```

This displays:
- Trace messages showing LLM calls
- Span information
- Final reward calculation

## Command-Line Options

```
--mode {generate-data,baseline,train,eval,debug}
                        Execution mode (default: baseline)
--limit LIMIT           Limit number of tasks for baseline (default: 50)
--iterations ITERATIONS Number of APO iterations (default: 10)
--prompt PROMPT         Path to optimized prompt file (default: outputs/best_prompt.txt)
--task TASK             Task ID for debug mode (default: 0)
```

## How It Works

### Architecture

The example implements a **light executor + heavy judge** pattern:

1. **Executor Agent**: Uses `gpt-4o-mini` to generate answers based on an optimizable prompt template
2. **Judge**: Uses `gpt-4o-mini` to evaluate answer quality with structured criteria:
   - Accuracy (0-1): Factual correctness
   - Completeness (0-1): Coverage of question aspects
   - Clarity (0-1): Readability and structure
3. **APO**: Automatically improves the prompt template based on judge feedback

### Cost Optimization

With 1000 training examples:
- Executor cost: 1000 × $0.0001 = $0.10
- Judge cost: 1000 × $0.002 = $2.00
- **Total: $2.10** (vs $30+ if using GPT-4 for both)

## Customization

### Use Your Own Dataset

Replace the generated data with your own:

```bash
# Prepare your data in JSONL format
cat > qa_train.jsonl << EOF
{"id": "custom_001", "question": "Your question?", "reference_answer": "Expected answer", "domain": "custom"}
EOF

# Run baseline
python qa_agent.py --mode baseline
```

### Modify Judge Criteria

Edit the `llm_judge()` function in `qa_agent.py` to customize evaluation criteria:

```python
judge_prompt = f"""Evaluate this answer on YOUR criteria:
1. Criterion A (0-1): Description
2. Criterion B (0-1): Description
...
"""
```

### Adjust APO Parameters

Modify `run_training()` in `qa_agent.py`:

```python
algo = agl.APO(
    openai_client,
    val_batch_size=20,      # More tasks per evaluation (more stable)
    gradient_batch_size=6,  # More gradients (better diversity)
    beam_width=3,           # More candidates (better exploration)
    branch_factor=3,        # More mutations (better coverage)
    beam_rounds=15,         # More iterations (better optimization)
)
```

## Related Documentation

- [Evaluate with LLM-as-Judge Recipe](../../docs/how-to/evaluate-with-llm-judge.md) - Complete guide with best practices
- [APO Algorithm](../../docs/algorithm-zoo/apo.md) - Technical details of Automatic Prompt Optimization
- [Training Your First Agent](../../docs/how-to/train-first-agent.md) - Introduction to Agent Lightning
