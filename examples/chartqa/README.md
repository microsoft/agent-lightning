# ChartQA Visual Reasoning Agent

Train vision-language models to reason about charts and graphs using multi-step workflows with self-refinement.

## Overview

This example demonstrates a **true multimodal agent** that:
- ✅ Analyzes charts visually (bar charts, line charts, pie charts, etc.)
- ✅ Performs multi-step reasoning (observe → extract → calculate → check → refine)
- ✅ Uses self-refinement loop (like SQL agent's check→rewrite pattern)
- ✅ Trains with VERL reinforcement learning
- ✅ Supports multiple vision-language models (Qwen2-VL, LLaMA-Vision, etc.)

## Architecture

```
START → analyze_chart → extract_data → calculate_answer
        → check_answer → [conditional]
           ├─→ END (if correct)
           └─→ refine_answer → extract_data → calculate_answer (loop back, max 3 turns)
```

This mirrors the SQL agent's proven `write→execute→check→rewrite` pattern, adapted for visual chart reasoning.

## Features

### Multi-Step Reasoning
1. **analyze_chart**: Observe chart type, axes, data series, patterns
2. **extract_data**: Extract specific values needed for the question
3. **calculate_answer**: Perform calculations and provide answer
4. **check_answer**: Verify answer for errors (extraction, calculation, logic)
5. **refine_answer**: Correct mistakes based on feedback (conditional loop)

### Self-Refinement
The agent can detect and correct its own mistakes:
- Misread values from chart → Re-extract correct values
- Arithmetic errors → Recalculate with correct operations
- Misunderstanding question → Adjust approach

### Training with VERL
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward**: Answer accuracy (exact match + numeric tolerance)
- **Model optimization**: Vision encoder + LLM decoder weights
- **Single GPU**: Optimized for RTX 4090 / A6000 (24GB VRAM)

## Setup

### 1. Install Dependencies

```bash
pip install datasets pillow pandas pyarrow tqdm
pip install langchain langgraph langchain-community
```

### 2. Download ChartQA Dataset

```bash
cd examples/chartqa
./download_chartqa.sh
```

This downloads the ChartQA dataset from HuggingFace (`HuggingFaceM4/ChartQA`) and saves images locally.

**Dataset Statistics:**
- Training: ~18,000 chart question-answer pairs
- Test: ~2,000 pairs
- Chart types: Bar, line, pie, scatter, etc.
- Sources: PlotQA, Statista charts
- **No HuggingFace token required** (public dataset)

### 3. Start vLLM Server

Start a vLLM server with a vision-language model:

```bash
# Option 1: Qwen2-VL-2B (recommended for 24GB GPU)
vllm serve Qwen/Qwen2-VL-2B-Instruct \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --port 8000

# Option 2: LLaMA-3.2-11B-Vision (requires more VRAM)
vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --port 8000
```

### 4. Set Environment Variables

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export MODEL=Qwen/Qwen2-VL-2B-Instruct
export CHARTQA_DATA_DIR=data
```

## Usage

### Debug Mode (Test Agent)

Test the agent on a few samples without training:

```bash
python chartqa_agent.py
```

This runs the agent on 5 test samples and prints detailed execution traces.

### Training

Train the agent with VERL reinforcement learning:

```bash
# Fast training (CI/testing, 1 epoch)
python train_chartqa_agent.py fast

# Standard Qwen2-VL-2B training (2 epochs)
python train_chartqa_agent.py qwen

# LLaMA-3.2-11B-Vision training (requires HF_TOKEN)
python train_chartqa_agent.py llama --active-agent my_agent
```

**Training Configuration:**
- **Batch size**: 4 (vision tokens are large)
- **Learning rate**: 1e-6 (lower for vision-language models)
- **FSDP offloading**: Enabled for 24GB GPU
- **Vision token caching**: Enabled for efficiency

## Example Execution

### Question: "What is the average of the three highest values?"

**Chart**: Bar chart showing 5 countries' GDP

```
Step 1: analyze_chart
<observe>
Bar chart showing GDP of 5 countries.
X-axis: Country names (USA, China, India, UK, France)
Y-axis: GDP in trillions USD
Values: USA ~25, China ~20, India ~15, UK ~10, France ~8
</observe>

Step 2: extract_data
<extract>
USA: 25, China: 20, India: 15, UK: 10, France: 8
</extract>

Step 3: calculate_answer
<calculate>
Three highest: 25, 20, 15
Average = (25 + 20 + 15) / 3 = 60 / 3 = 20
</calculate>
<answer>
20
</answer>

Step 4: check_answer
Review: All values correctly extracted. Math is correct: (25+20+15)/3 = 20.
THE ANSWER IS CORRECT.

→ END (no refinement needed)
```

### Example with Refinement

```
Step 1-3: Initial attempt with arithmetic error
<answer>19.5</answer>  # Wrong! Divided by 4 instead of 3

Step 4: check_answer
Error detected: Calculation used wrong divisor (4 instead of 3).
THE ANSWER IS INCORRECT.

Step 5: refine_answer
<calculate>
Corrected: (25 + 20 + 15) / 3 = 20
</calculate>
<answer>
20
</answer>

Step 6: check_answer (second iteration)
THE ANSWER IS CORRECT.

→ END
```

## Comparison with VQA Agent

| Feature | VQA Agent | ChartQA Agent |
|---------|-----------|---------------|
| **Task** | General image QA | Chart reasoning |
| **Steps** | 1 (single-turn) | 5+ (multi-turn) |
| **Tools** | None | Data extraction, calculator |
| **Self-correction** | ❌ No | ✅ Yes (refinement loop) |
| **Agenticness** | Low | **High** |
| **Training benefit** | Marginal | **Significant** (tool selection, reasoning) |

## Configuration

### Model Options

**Qwen2-VL Series** (Recommended):
- `Qwen/Qwen2-VL-2B-Instruct` - Best for 24GB GPU
- `Qwen/Qwen2-VL-7B-Instruct` - Better accuracy, needs 40GB+

**LLaMA Vision**:
- `meta-llama/Llama-3.2-11B-Vision-Instruct` - Requires HF_TOKEN

**Other Models** (via vLLM):
- `llava-hf/llava-v1.6-vicuna-7b-hf`
- `OpenGVLab/InternVL2_5-2B`

### Training Configs

Edit `train_chartqa_agent.py` `RL_TRAINING_CONFIG`:

```python
"data": {
    "train_batch_size": 4,  # Reduce if OOM
    "max_prompt_length": 4096,  # Adjust for longer charts
}
"actor_rollout_ref": {
    "rollout": {
        "gpu_memory_utilization": 0.6,  # Adjust based on VRAM
    },
    "actor": {
        "optim": {"lr": 1e-6},  # Learning rate
    }
}
```

## Dataset Format

**Parquet Schema:**
```python
{
    "id": "train_0",
    "image_path": "train/png/two_col_12345.png",  # Relative to data/
    "question": "What is the average of the three highest values?",
    "answer": "20"
}
```

## Evaluation

**Reward Computation:**
- **Exact match**: 1.0 (pred == ground_truth)
- **Numeric tolerance**: 1.0 (within 2% relative error)
- **Partial match**: 0.5 (substring overlap)
- **No match**: 0.0

## Troubleshooting

### vLLM Out of Memory
- Reduce `gpu_memory_utilization` to 0.5
- Use smaller model (Qwen2-VL-2B instead of 7B)
- Reduce `max_model_len`

### Training OOM
- Reduce `train_batch_size` to 2
- Enable more offloading in `fsdp_config`
- Use gradient checkpointing

### Image Not Found Errors
- Check `CHARTQA_DATA_DIR` points to correct location
- Verify images extracted: `ls data/train/png/`
- Use absolute paths

### Agent Not Refining
- Check `max_turns` in `LitChartQAAgent`
- Verify `CHECK_ANSWER_PROMPT` feedback includes "INCORRECT"
- Enable debug mode: `debug=True` in `ChartQAAgent`

## Dataset Source

**HuggingFace**: `HuggingFaceM4/ChartQA`
- Public dataset (no token required)
- URL: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
- Original paper: [ChartQA (ACL 2022)](https://aclanthology.org/2022.findings-acl.177/)

## Citation

**ChartQA Dataset:**
```bibtex
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed and Long, Do and Tan, Jia Qing and Joty, Shafiq and Hoque, Enamul",
    booktitle = "Findings of ACL 2022",
    year = "2022",
}
```

**AgentLightning:**
```bibtex
@software{agentlightning2024,
    title = {Agent Lightning: Scalable RL Training for Agentic AI},
    author = {{Microsoft Research}},
    year = {2024},
    url = {https://github.com/microsoft/agent-lightning}
}
```

## License

This example follows ChartQA's MIT License. See the [HuggingFace dataset page](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) for details.
