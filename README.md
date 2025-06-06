# Agent Lightning

## Installation

The `/path/to/agentlightning` refers to the directory where this README file is located.

1. We recommend creating a new virtual / conda environment with Python 3.10 or higher.
2. Install uv (required for some MCP agents): `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install PyTorch, FlashAttention and VLLM. The following versions and setup orders are tested to work.
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   pip install flash-attn --no-build-isolation
   pip install vllm==v0.8.5.post1
   ```
4. Install the patched VERL. You must use the commit below and install our patch to ensure compatibility. Uninstall any existing VERL first.
   ```bash
   git clone https://github.com/volcengine/verl /path/to/your/verl
   cd /path/to/your/verl
   git checkout 2dc3e0ebadb479bb3f2b48cfc7f28a3b70d5ce60
   pip install -e .
   cd /path/to/agentlightning
   bash scripts/verl_git_apply.sh /path/to/your/verl
   ```
5. Install AgentLightning.
   ```bash
   cd /path/to/agentlightning
   pip install -e .
   ```
6. Install Agent frameworks and utils (optional). You can skip this step if you don't need them.
   ```bash
   # Install AutoGen (recommended to do this first)
   pip install "autogen-agentchat" "autogen-ext[openai]"

   # Install LiteLLM
   pip install "litellm[proxy]"
   
   # Install MCP
   pip install mcp

   # Install OpenAI Agents
   pip install openai-agents

   # Install LangChain
   pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters

   # Install SQL-related dependencies
   pip install sqlparse nltk
   ```

## Quickstart

The core of Agent Lightning consists of two parts: one training server and one to mulitple agents.
The server is responsible for iterating over the data, preparing the sample, and providing the LLM endpoint.
The agent retrieves from the sample queue, processing the sample (optionally iteracting with the provided LLM endpoint), and sending the result back to the server.
The results are list of prompts and responses sent and received from the LLM endpoints.
The server collects the result (aka trajectories) and computes the loss to optimize the LLMs.

See the `examples` folder for more complete examples.

*Coming soon:* a short demonstration code snippet.

## General Caveats

1. Our current implementation uses [AgentOps](https://github.com/AgentOps-AI/agentops) to track the agents. By default, the usage of AgentOps (init and trace) is managed by AgentLightning. If you have used AgentOps in your own code, you will need to switch of `agentops_managed` in Trainer and figure out the integration yourself.
2. As the trace can sometimes be problematic, you can use `processor.last_trace()._tree_visualize("tree_graph")` to verify whether the trace is correct. Note that this API is not stable and may change in the future.
3. The current implementation does not support launching agent clients and training server in one command. You will need to launch them in two separate terminals or launch one of them in the background. The order does not matter in most cases.
4. The environment variables and working directory of `ray init` matters. In case of file not found errors, try to restart the ray in your current working directory.
5. Currently the training server will hang when some samples fail or timeout on the agent side. Therefore, it's always recommended to limit the prompt length and response length to avoid such issues, as it accounts for most of these failures.
