# Agent Lightning

Welcome to Agent Lightning! Agent Lightning is a flexible and extensible training framework that enables seamless model training for any existing agent framework.

This guide will walk you through setting up and running the project.

## Installation

First, let's get your environment set up. We'll be using `/path/to/agentlightning` to refer to the directory containing this README file.

### 1. Set Up Your Environment

We strongly recommend creating a new virtual environment to avoid conflicts with other packages. You can use either `conda` or `venv`. **Python 3.10 or later** is recommended.

### 2. Install Core Dependencies

Next, let's install the essential packages: `uv`, `PyTorch`, `FlashAttention`, and `vLLM`.

  * **Install `uv`** (This is required for some MCP agents; otherwise some agents might hang):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

  * **Install `PyTorch`, `FlashAttention`, and `vLLM`**:
    The following versions and installation order have been tested and are confirmed to work.

    ```bash
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install flash-attn --no-build-isolation
    pip install vllm==0.9.2
    ```

### 3. Install VERL

Agent Lightning requires VERL for RL training. Please install the latest version from main branch.

```bash
git clone https://github.com/volcengine/verl /path/to/your/verl
cd /path/to/your/verl
pip install -e .
```

### 4. Install Agent Lightning

Now, you're ready to install Agent Lightning itself.

```bash
cd /path/to/agentlightning
pip install -e .
```

### 5. Install Optional Frameworks

If you plan to use other agent frameworks, you can install them with the following commands. If you don't need these, feel free to skip this step.

```bash
# AutoGen (Recommended to install first)
pip install "autogen-agentchat" "autogen-ext[openai]"

# LiteLLM
pip install "litellm[proxy]"

# MCP
pip install mcp

# OpenAI Agents
pip install openai-agents

# LangChain
pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters

# SQL-related dependencies
pip install sqlparse nltk
```

Don't worry if dependency conflicts arise during this step. Follow the installation order above and the conflicts generally do not matter.

## Architecture

Currently, Agent Lightning is built around a **training server** and one or multiple **agents**.

  * The **server** manages the training data, prepares samples for the agents, and provides the LLM endpoint.
  * **Agents** retrieve samples from the server, process them (which may involve interacting with the LLM), and send the results back. These results, or "trajectories," are lists of prompts and responses from the LLM.
  * The **server** then collects these trajectories and computes the loss to optimize the language models.

## Development Instructions

Install with development dependencies:

```
pip install -e .[dev]
```

Please run pre-commit hooks before checking in code:

```
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

## Examples

For more detailed examples, please see the `examples` folder.

## Important Caveats

1.  **AgentOps Integration**: Agent Lightning uses [AgentOps](https://github.com/AgentOps-AI/agentops) for agent tracking by default. If you're already using AgentOps in your own code, you'll need to disable our managed AgentOps client by modifying the `tracer` parameter of trainer.

2.  **Debugging Traces**: If you encounter issues with tracing, you can visualize the trace tree using `tracer.last_trace().visualize("tree_graph")`. Please note that this API is experimental and may change in future releases.

3.  **Launching the Server and Agents**: Currently, the training server and agent clients must be launched in separate processes. You can open two terminal windows or run one of them in the background. The order in which you launch them generally doesn't matter.

4.  **Environment Variables**: The environment variables and working directory at the time of `ray init` are important. If you run into "file not found" errors, try restarting Ray from your current working directory.

5.  **Handling Timeouts**: The training server may hang if samples fail or time out on the agent side. To prevent this, we recommend setting limits on the prompt and response lengths, as this is the most common cause of failures.

6.  **VERL Failures**: Save checkpoints frequently, as VERL with vLLM may sometimes experience out-of-memory issues. If you encounter a VERL failure, you can resume training from the last checkpoint.
