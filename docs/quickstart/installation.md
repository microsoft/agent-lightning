# Installation

## Install from PyPI

### Set Up Your Environment

We strongly recommend creating a new virtual environment to avoid conflicts with other packages. You can use either `conda` or `venv`. **Python 3.10 or later** is recommended.

### Install Core Training Dependencies (Optional)

If you are running RL with Agent-Lightning, the next step is to install the essential packages: `PyTorch`, `FlashAttention`, `vLLM` and `VERL`. The following versions and installation order have been tested and are confirmed to work.

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install vllm==0.9.2
pip install verl==0.5.0
```

See [this script]({{ config.repo_url }}/tree/{{ config.extra.source_commit }}/scripts/setup_stable_gpu.sh) for a full installation script.

### Install Agent Lightning

Now, you're ready to install Agent Lightning itself.

```bash
pip install agentlightning
```

### Install Agent Frameworks (Optional)

If you plan to use other agent frameworks, you can install them with the following commands. If you don't need these, feel free to skip this step.
We recommend doing this as the final step to avoid dependency versions being overwritten by mistake.

```bash
# AutoGen (Recommended to install first)
pip install "autogen-agentchat" "autogen-ext[openai]"

# LiteLLM
pip install "litellm[proxy]"

# MCP
pip install mcp

# UV
pip install uv

# OpenAI Agents
pip install openai-agents

# LangChain
pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters

# SQL-related dependencies
pip install sqlparse nltk
```

### Shortcuts for installing Extra Dependencies

For development:
```bash
pip install agentlightning[dev]
```

For agent support:
```bash
pip install agentlightning[agent]
```

## Install from Source

```
git clone {{ config.repo_url }}
cd agent-lightning
pip install -e .[dev]
```

Please run pre-commit hooks before checking in code:

```
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```
