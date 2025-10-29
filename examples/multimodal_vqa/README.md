# Multimodal Visual Question Answering (VQA) with VERL Training

[![Issue #105](https://img.shields.io/badge/Issue-105-blue)](https://github.com/microsoft/agent-lightning/issues/105)

This example demonstrates **end-to-end reinforcement learning for multimodal vision-language models** using Agent Lightning and VERL (Versatile Reinforcement Learning).

## 🌟 Key Features

- ✅ **Local Model Training**: Fine-tune Qwen2-VL with VERL/GRPO algorithm
- ✅ **Vision Token Processing**: End-to-end training with visual embeddings
- ✅ **Single GPU Optimization**: Runs on 24GB GPU (RTX 4090/A6000)
- ✅ **Reward-Based Learning**: Optimize for VQA accuracy using custom rewards
- ✅ **Distributed Rollout**: Scale with LLM proxy for parallel task execution
- ✅ **Baseline Comparison**: Compare RL-trained model with API baselines

## 📋 What's New in RL Version

**Previous Approach (API-based with APO):**
- Used GPT-4V/Claude via APIs
- APO only optimized prompt text
- No model weight updates
- High API costs

**New Approach (RL-based with VERL):**
- Local Qwen2-VL model training
- VERL/GRPO updates model parameters
- Vision tokens processed locally
- Full model fine-tuning
- Cost-effective with GPU

## 🏗️ Architecture

```
VQA Task → vLLM (Qwen2-VL) → Agent → Reward → VERL → Model Update
    ↓           ↓                       ↑
  Image    Vision Encoder          Answer Quality
```

**Components:**
1. **Data**: VQAv2 dataset with COCO images
2. **Model**: Any vision-language model (Qwen-VL, LLaVA, InternVL, etc.)
3. **Inference**: vLLM or SGLang for fast multimodal inference
4. **Agent**: `@agl.rollout` decorated function with reward emission
5. **Training**: VERL with GRPO algorithm (auto-manages vLLM/SGLang)
6. **Optimization**: FSDP with offloading for 24GB GPU

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- GPU: RTX 4090, A6000, or equivalent (24GB VRAM minimum)
- Storage: ~20GB for VQAv2 dataset

**Software:**
```bash
# Install Agent Lightning with VERL dependencies
# Note: VERL installation requires GPU and CUDA
cd agent-lightning
pip install -e ".[verl]"

# Or install manually
pip install vllm>=0.8.4 verl>=0.5.0 transformers torch
```

### Dataset Setup

```bash
cd examples/multimodal_vqa

# 1. Download VQAv2 dataset (~20GB)
bash download_vqav2.sh
# Note: Edit the script to uncomment download commands

# 2. Convert to parquet format
pip install pandas pyarrow

python data_utils.py convert \
    --questions data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
    --annotations data/vqav2/v2_mscoco_train2014_annotations.json \
    --image-dir data/vqav2/images/train2014 \
    --output data/vqav2_train.parquet

python data_utils.py convert \
    --questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations data/vqav2/v2_mscoco_val2014_annotations.json \
    --image-dir data/vqav2/images/val2014 \
    --output data/vqav2_val.parquet

# 3. Verify data
python data_utils.py verify --file data/vqav2_train.parquet
```

### Training

**Debug Mode (Quick Test):**
```bash
# Test agent with single task
python vqa_agent.py
```

**Local Training:**
```bash
# Train with 4 parallel runners
python train_vqa_agent.py --n-runners 4 --max-steps 500
```

**Distributed Training (Recommended):**
```bash
# With LLM proxy for distributed rollout
python train_vqa_agent.py --llm-proxy --n-runners 10 --max-steps 1000
```

**Custom Configuration:**
```bash
python train_vqa_agent.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --data-path data/vqav2_train.parquet \
    --val-data-path data/vqav2_val.parquet \
    --n-runners 8 \
    --max-steps 1000 \
    --llm-proxy
```

**Try Different Models:**
```bash
# LLaVA (both vLLM and SGLang)
python train_vqa_agent.py --model llava-hf/llava-v1.6-vicuna-7b-hf

# InternVL (vLLM only)
python train_vqa_agent.py --model OpenGVLab/InternVL2_5-2B

# Llama Vision (both vLLM and SGLang)
python train_vqa_agent.py --model meta-llama/Llama-3.2-11B-Vision-Instruct

# Janus-Pro (SGLang only - image understanding + generation)
python train_vqa_agent.py --model deepseek-ai/Janus-Pro-1B
```

### Baseline Comparison

```bash
# Run API baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python vqa_agent_baseline.py
```

## 📂 Project Structure

```
multimodal_vqa/
├── README.md                    # This file
├── vqa_agent.py                 # VERL agent with @agl.rollout
├── vqa_agent_baseline.py        # OpenAI API baseline for comparison
├── train_vqa_agent.py           # VERL training script
├── data_utils.py                # VQAv2 data processing utilities
├── download_vqav2.sh            # Dataset download script
└── data/
    ├── vqav2/                   # Raw VQAv2 data
    │   ├── images/
    │   │   ├── train2014/
    │   │   └── val2014/
    │   └── *.json               # Annotations and questions
    ├── vqav2_train.parquet      # Processed training data
    └── vqav2_val.parquet        # Processed validation data
```

## 🔧 Technical Details

### VERL Configuration

The training uses VERL (Versatile Reinforcement Learning) with GRPO (Group Relative Policy Optimization).

**Key Features:**
- VERL automatically manages vLLM or SGLang servers
- You specify the model, VERL handles the serving framework
- Both frameworks support vision-language models

```python
verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",  # GRPO algorithm
    },
    "data": {
        "train_batch_size": 16,
        "max_prompt_length": 2048,  # Increased for vision tokens
        "max_response_length": 512,
    },
    "actor_rollout_ref": {
        "rollout": {
            "name": "vllm",  # VERL manages vLLM/SGLang automatically
            "gpu_memory_utilization": 0.6,
            "enable_prefix_caching": True,  # Cache vision tokens for efficiency
        },
        "actor": {
            "fsdp_config": {
                "param_offload": True,      # CPU offload for 24GB GPU
                "optimizer_offload": True,  # Optimizer offload
            },
            "optim": {"lr": 1e-6},  # Lower LR for vision-language models
        },
    },
}
```

**Single GPU Optimizations:**
- FSDP (Fully Sharded Data Parallel) with parameter/optimizer offloading
- Smaller batch size (16) to fit in 24GB VRAM
- Gradient checkpointing for memory efficiency
- GPU memory utilization: 0.6 (leaves room for training)
- Works with models up to 7B parameters

### Vision Token Processing

**How vision-language models process images:**

1. **Image Loading**: Local file loaded via `file://` protocol
2. **Vision Encoding**: Image → Vision Encoder → Visual Embeddings (typically 256-576 tokens)
3. **Projection**: Visual embeddings projected to LLM dimension
4. **Concatenation**: Vision tokens + Text tokens → Unified sequence
5. **LLM Processing**: Standard transformer processing on combined tokens
6. **Generation**: Text output based on visual + text context

**Vision Token Counts (Approximate):**
- Qwen2-VL: 256-576 tokens per image
- LLaVA: 576 tokens per image
- InternVL: 256 tokens per image
- Llama-Vision: 1024+ tokens per image (higher resolution)

**Example message format (OpenAI-compatible):**
```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
    ]
}
```

**Note**: Both vLLM and SGLang support this OpenAI-compatible message format for multimodal inputs.

### Reward Function

The agent uses a simple similarity-based reward:

```python
def compute_answer_similarity(predicted: str, expected: str) -> float:
    """
    - Exact match: 1.0
    - Contains match: 0.8
    - Word overlap: proportional (0.0-1.0)
    """
```

**Improvement Ideas:**
- Use VQAv2 official evaluation metric
- Implement LLM-as-judge for semantic similarity
- Multi-criteria rewards (accuracy + brevity)

## 📊 Dataset Format

### Parquet Schema

```
id: str           # Question ID (e.g., "262148000")
image_path: str   # Absolute path to COCO image
question: str     # Question about the image
answer: str       # Most common answer from 10 annotators
```

### Example Tasks

```json
{
  "id": "262148000",
  "image_path": "/path/to/COCO_train2014_000000262148.jpg",
  "question": "What color is the cat?",
  "answer": "black"
}
```

## 🎯 Model Support

Agent Lightning's VERL integration supports both **vLLM** and **SGLang** serving frameworks for vision-language models.

### Serving Frameworks

| Framework | Status | Notes |
|-----------|--------|-------|
| **vLLM** | ✅ Recommended | Deep Agent Lightning integration, production-ready, best token ID support |
| **SGLang** | ✅ Supported | Alternative option, supports more recent models, managed by VERL |

**Note**: VERL automatically manages the serving framework. You specify the model, and VERL handles vLLM/SGLang configuration.

### Supported Vision-Language Models

#### Common Models (Both vLLM and SGLang)

These models work with both frameworks and are **recommended** for maximum compatibility:

| Model Family | Example Models | Parameters | VRAM (24GB GPU) | VRAM (40GB+ GPU) |
|--------------|----------------|------------|-----------------|------------------|
| **Qwen-VL** | Qwen2-VL-2B/7B, Qwen2.5-VL, Qwen3-VL | 2B-7B | ✅ 2B only | ✅ All |
| **LLaVA** | LLaVA-v1.5/v1.6-7B, LLaVA-NeXT, LLaVA-OneVision | 7B-34B | ✅ 7B only | ✅ All |
| **Llama Vision** | Llama-3.2-11B-Vision-Instruct | 11B | ⚠️ With offloading | ✅ Yes |
| **DeepSeek-VL2** | DeepSeek-VL2-1.6B/2.7B | 1.6B-2.7B | ✅ Yes | ✅ Yes |
| **MiniCPM-V** | MiniCPM-V-2.6, MiniCPM-o | 8B | ✅ With offloading | ✅ Yes |
| **Phi-4** | Phi-4-multimodal-instruct | 5.6B | ✅ Yes | ✅ Yes |
| **GLM-4V** | GLM-4V-9B, GLM-4.5V | 9B | ⚠️ With offloading | ✅ Yes |

#### vLLM-Only Models

Additional models supported exclusively by vLLM:

| Model Family | Example Models | Parameters | Best GPU |
|--------------|----------------|------------|----------|
| **InternVL** | InternVL2.5-2B, InternVL3-8B, InternVL3.5 | 2B-8B | 24GB+ |
| **Llama-4-Vision** | Llama-4-Vision (upcoming) | TBD | TBD |
| **Pixtral** | Pixtral-12B | 12B | 40GB+ |
| **IDEFICS3** | IDEFICS3-8B-Llama3 | 8B | 24GB+ |
| **Molmo** | Molmo-7B-D | 7B | 24GB+ |

#### SGLang-Only Models

Additional models supported exclusively by SGLang:

| Model Family | Example Models | Parameters | Best GPU |
|--------------|----------------|------------|----------|
| **Janus-Pro** | Janus-Pro-1B/7B (image understanding + generation) | 1B-7B | 24GB+ |
| **Gemma 3 Multimodal** | Gemma-3-4B/12B/27B-MM | 4B-27B | 40GB+ |
| **Kimi-VL** | Kimi-VL-A3B | 3B | 24GB |
| **NVILA** | NVILA-8B/15B, NVILA-Lite | 8B-15B | 40GB+ |
| **MiMo-VL** | MiMo-VL-7B | 7B | 24GB+ |
| **DotsVLM** | DotsVLM | TBD | 24GB+ |

### Model Selection Guide

**For 24GB GPU (RTX 4090/A6000):**
- **Recommended**: Qwen2-VL-2B, DeepSeek-VL2-1.6B, InternVL2.5-2B
- Enable FSDP offloading
- Batch size: 16
- Models up to 7B parameters work with aggressive offloading

**For 40GB+ GPU (A100/H100):**
- **Recommended**: Qwen2-VL-7B, LLaVA-NeXT-34B, Llama-3.2-11B-Vision
- Disable offloading for speed
- Batch size: 32
- Can train models up to 34B parameters

**For Multi-GPU:**
- Scale up to larger models (34B+)
- Disable offloading
- Increase batch size proportionally
- Use more runners for parallel rollout

### Framework-Specific Notes

**vLLM:**
- Better token ID support (critical for training stability)
- More mature, production-ready
- Default port: 8000
- Start server: `vllm serve Qwen/Qwen2-VL-2B-Instruct --port 8000`

**SGLang:**
- Supports newer models faster
- Efficient KV cache for multimodal inputs
- Default port: 30000
- Start server: `python -m sglang.launch_server --model Qwen/Qwen2-VL-2B-Instruct --port 30000`
- Use `--keep-mm-feature-on-device` flag for lower latency (higher VRAM usage)

### Complete Model Lists

For the most up-to-date lists:
- vLLM: https://docs.vllm.ai/en/latest/models/supported_models.html
- SGLang: https://docs.sglang.ai/supported_models/multimodal_language_models

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Solution 1: Reduce batch size
python train_vqa_agent.py --n-runners 2  # Fewer parallel tasks

# Solution 2: Lower vLLM memory utilization
# Edit train_vqa_agent.py: gpu_memory_utilization=0.5
```

**2. CUDA/GPU Not Detected**
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**3. vLLM Import Error**
```bash
# VERL requires specific versions
pip install "vllm>=0.8.4,<0.11.0" "verl>=0.5.0,<0.6.0"
```

**4. Image Not Found Errors**
```bash
# Verify image paths in parquet
python data_utils.py verify --file data/vqav2_train.parquet

# Check COCO images exist
ls data/vqav2/images/train2014/ | head
```

**5. Slow Training**
```bash
# Use distributed rollout with proxy
python train_vqa_agent.py --llm-proxy --n-runners 10

# Enable prefix caching (already enabled in config)
# This caches vision token embeddings
```

## 🔬 Evaluation

**Compare with Baseline:**
```bash
# 1. Train RL model
python train_vqa_agent.py --max-steps 1000

# 2. Evaluate on validation set
# (Evaluation is automatically done during training)

# 3. Compare with API baseline
python vqa_agent_baseline.py
```

**Metrics to Track:**
- Average reward on validation set
- Accuracy (exact match)
- Training time
- GPU memory usage

## 🚀 Advanced Usage

### Custom Dataset

Create your own parquet file:

```python
import pandas as pd

data = [
    {
        "id": "custom_001",
        "image_path": "/path/to/image1.jpg",
        "question": "What is this?",
        "answer": "a cat"
    },
    # ... more tasks
]

df = pd.DataFrame(data)
df.to_parquet("data/custom_vqa.parquet")
```

Then train:
```bash
python train_vqa_agent.py \
    --data-path data/custom_vqa.parquet \
    --val-data-path data/custom_vqa_val.parquet
```

### Multi-GPU Training

```bash
# Use torchrun for multi-GPU
torchrun --nproc_per_node=4 train_vqa_agent.py \
    --n-runners 16 \
    --max-steps 2000
```

### Distributed Workers

**On training server:**
```bash
# Start trainer with proxy
python train_vqa_agent.py --llm-proxy --n-runners 20
# Proxy will start on port 8080
```

**On worker machines:**
```bash
# Point to training server's proxy
export OPENAI_BASE_URL=http://training-server:8080/v1
python worker_script.py  # Your worker implementation
```

## 📖 Additional Resources

**Agent Lightning:**
- [Issue #105 Discussion](https://github.com/microsoft/agent-lightning/issues/105)
- [Agent Lightning Documentation](https://agent-lightning.readthedocs.io/)
- [VERL Paper](https://arxiv.org/abs/2402.03081)
- [Serving LLMs Guide](https://agent-lightning.readthedocs.io/deep-dive/serving-llm.html)

**Serving Frameworks:**
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang Multimodal Models](https://docs.sglang.ai/supported_models/multimodal_language_models)

**Models:**
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [LLaVA Models](https://huggingface.co/llava-hf)
- [InternVL Models](https://huggingface.co/OpenGVLab)
- [Llama Vision](https://huggingface.co/meta-llama)

**Dataset:**
- [VQAv2 Dataset](https://visualqa.org/)

## 🤝 Contributing

This example is part of [Issue #105](https://github.com/microsoft/agent-lightning/issues/105). Contributions welcome for:

- [ ] Implementing VQAv2 official evaluation metrics
- [ ] Adding LLM-as-judge reward function
- [ ] Supporting more vision-language models (LLaVA, InternVL)
- [ ] Multi-image VQA tasks
- [ ] Video question answering

## 📝 License

Copyright (c) Microsoft. All rights reserved.

Licensed under the MIT License.

## 🙏 Acknowledgments

- Agent Lightning team for the framework
- VERL team for the RL algorithm
- Qwen team for Qwen2-VL model
- VQAv2 dataset creators
