# Instruction-Pretrain

This dataset is intended for training general context-based task completion (i.e., reading comprehension) abilities using the [seed data in Instruction Pre-Training](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection).

**NOTE**: This is used for LLM-in-Sandbox-RL training, not for testing.

## Dependencies

```bash
pip install math-verify rouge-score
```

## Question Types

The dataset contains three types of questions:
- **Multiple choice (single answer)**: Select one correct option (A, B, C, or D)
- **Multiple choice (multiple answers)**: Select all correct options (e.g., "A, C, D")
- **Open-ended**: Free-form text answer

## Scoring

- **Single choice**: Exact match (1.0 or 0.0)
- **Multiple choice**: F1 score based on precision and recall
- **Open-ended**: 
  1. First try math verification (for numerical answers)
  2. Fall back to ROUGE-L score

## Answer Format

The agent should save answers to `/testbed/output/answer.txt`:
- Single choice: `A`
- Multiple choice: `A, C, D`
- Open-ended: Free text

## Dataset

Using HuggingFace dataset: `daixuancheng/llm-in-sandbox-rl` (config: `instruct_pretrain`)

## Usage

Please refer to our RL code for usage. You may also use it for validation or debugging:
```bash
# LLM-in-Sandbox
llm-in-sandbox benchmark --task instruct_pretrain --start_id 0 --end_id 3

# Vanilla LLM
llm-in-sandbox benchmark --task instruct_pretrain --mode llm --start_id 0 --end_id 3
```