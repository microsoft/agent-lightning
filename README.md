<div style="text-align:center; margin-bottom:20px;">
  <img src="docs/assets/readme-banner.svg" alt="Agent-lightning-banner" style="max-width:600px"/>
</div>

# Agent Lightning⚡

[![CPU Test](https://github.com/microsoft/agent-lightning/actions/workflows/tests.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/tests.yml)
[![GPU Test](https://github.com/microsoft/agent-lightning/actions/workflows/examples.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples.yml)
[![PyPI version](https://badge.fury.io/py/agentlightning.svg)](https://badge.fury.io/py/agentlightning)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/RYk7CdvDR7)

**The absolute trainer to light up AI agents.**

Join our [Discord community](https://discord.gg/RYk7CdvDR7) to connect with other users and contributors.

## ⚡ Core Features

- Turn your agent into an optimizable beast with **ZERO CODE CHANGE** (almost)! 💤
- Build with **ANY** agent framework (LangChain, OpenAI Agent SDK, AutoGen, CrewAI, ...); or even WITHOUT agent framework (Python OpenAI). You name it! 🤖
- **Selectively** optimize one or more agents in a multi-agent system. 🎯
- Embraces Reinforcement Learning, Automatic Prompt Optimization and more **algorithms**. 🤗

![Agent-Lightning-code-diff](docs/assets/readme-diff.png)

## ⚡ Resources

- 8/11/2025 [Training AI Agents to Write and Self-correct SQL with Reinforcement Learning](https://medium.com/@yugez/training-ai-agents-to-write-and-self-correct-sql-with-reinforcement-learning-571ed31281ad) Medium.
- 8/5/2025 [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680) arXiv paper.
- 7/26/2025 [We discovered an approach to train any AI agent with RL, with (almost) zero code changes.](https://www.reddit.com/r/LocalLLaMA/comments/1m9m670/we_discovered_an_approach_to_train_any_ai_agent/) Reddit.
- 6/6/2025 [Agent Lightning - Microsoft Research](https://www.microsoft.com/en-us/research/project/agent-lightning/) Project page.

## ⚡ Community Projects

- [DeepWerewolf](https://github.com/af-74413592/DeepWerewolf) — A case study of agent RL training for the Chinese Werewolf game built with AgentScope and Agent Lightning.
- [AgentFlow](https://agentflow.stanford.edu/) — A modular multi-agent framework that combines planner, executor, verifier, and generator agents with the Flow-GRPO algorithm to tackle long-horizon, sparse-reward tasks.

## ⚡ Installation

First, let's get your environment set up. We'll be using `/path/to/agentlightning` to refer to the directory containing this README file.

### 1. Set Up Your Environment

We strongly recommend creating a new virtual environment to avoid conflicts with other packages. You can use either `conda` or `venv`. **Python 3.10 or later** is recommended.

### 2. Install Core Training Dependencies (Optional)

If you are running RL with Agent-Lightning, the next step is to install the essential packages: `PyTorch`, `FlashAttention`, `vLLM` and `VERL`. The following versions and installation order have been tested and are confirmed to work.

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install vllm==0.9.2
pip install verl==0.5.0
```

See `scripts/setup_stable_gpu.sh` for a full installation script.

### 3. Install Agent Lightning

Now, you're ready to install Agent Lightning itself.

```bash
pip install agentlightning
```

### 4. Install Agent Frameworks (Optional)

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

Don't worry if dependency conflicts arise during this step. Follow the installation order above and the conflicts generally do not matter.

## ⚡ Examples

For more detailed examples, please see the `examples` folder:

1. [calc_x](examples/calc_x): An agent built with AutoGen with calculator tool use, trained on Calc-X dataset with Reinforcement Learning.
2. [spider](examples/spider): A write-check-rewrite looped agent with LangGraph with SQL execution; selectively optimize write and rewrite on Spider dataset with Reinforcement Learning.
3. [apo](examples/apo): An example to customize an optimization algorithm: Automatic Prompt Optimization.

## ⚡ Important Caveats

1.  **AgentOps Integration**: Agent Lightning uses [AgentOps](https://github.com/AgentOps-AI/agentops) for agent tracking by default. If you're already using AgentOps in your own code, you'll need to disable our managed AgentOps client by modifying the `tracer` parameter of trainer.
2.  **Debugging Traces**: If you encounter issues with tracing, you can visualize the trace tree using `tracer.last_trace().visualize("tree_graph")`. Please note that this API is experimental and may change in future releases.
3.  **Launching the Server and Agents**: Currently, the training server and agent clients must be launched in separate processes. You can open two terminal windows or run one of them in the background. The launching order generally doesn't matter.
4.  **Environment Variables**: The environment variables and working directory at the time of `ray init` are important. If you run into "file not found" errors, try restarting Ray from your current working directory.
5.  **Handling Timeouts**: The training server may hang if samples fail or time out on the agent side. To prevent this, we recommend setting limits on the prompt and response lengths, as this is the most common cause of failures.
6.  **VERL Failures**: Save checkpoints frequently, as VERL with vLLM may sometimes experience out-of-memory issues. If you encounter a VERL failure, you can resume training from the last checkpoint.

## ⚡ Architecture

Currently, Agent Lightning is built around a **training server** and one or multiple **agents**.

* The **server** manages the training data, prepares samples for the agents, and provides the LLM endpoint.
* **Agents** retrieve samples from the server, process them (which may involve interacting with the LLM), and send the results back. These results, or "trajectories," are lists of prompts and responses from the LLM.
* The **server** then collects these trajectories and computes the losses to optimize the language models.

![Agent-Lightning-architecture](docs/assets/readme-architecture.png)

## ⚡ Development Instructions

Install with development dependencies:

```
git clone https://github.com/microsoft/agent-lightning
cd agent-lightning
pip install -e .[dev]
```

Please run pre-commit hooks before checking in code:

```
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

Serve documentation locally:

```bash
mkdocs serve
```

## ⚡ Citation

If you find Agent Lightning useful in your research or projects, please cite our paper:

```bibtex
@misc{luo2025agentlightningtrainai,
      title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
      author={Xufang Luo and Yuge Zhang and Zhiyuan He and Zilong Wang and Siyun Zhao and Dongsheng Li and Luna K. Qiu and Yuqing Yang},
      year={2025},
      eprint={2508.03680},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.03680},
}
```

## ⚡ Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## ⚡ Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## ⚡ Responsible AI

This project has been evaluated and certified to comply with the Microsoft Responsible AI Standard. The team will continue to monitor and maintain the repository, addressing any severe issues, including potential harms, if they arise.

## ⚡ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
