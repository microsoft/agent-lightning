<p align="center">
  <img src="docs/images/agl-v1.0.jpg" alt="Agent Lightning v1.0" width="500">
</p>

<p align="center"><em>3,500-Line Lightweight Agentic RL Framework for Training Agents with Real Harnesses!</em></p>

<p align="center">
  <a href="https://microsoft.github.io/agent-lightning/stable/">Documentation</a> &nbsp;·&nbsp; Technical Report (Coming Soon) &nbsp;·&nbsp; <a href="LICENSE">MIT License</a>
</p>

> Agent Lightning was completely refactored in v1.0. For legacy releases earlier than v1.0, see [this branch](https://github.com/microsoft/agent-lightning/tree/v0.x).

## ⚡ Key Features

- 🪶 **~3,500 lines of core Python:** We treat simplicity as the first principle.
- 🧩 **Train with real agent harnesses:** Agents interact with the model through the Agent Lightning v1.0 proxy with **ZERO changes**, while keeping tools, context, control flow, and environments in the loop.
- ☸️ **Native Kubernetes support:** Run agents directly as Kubernetes Jobs without relying on external sandbox services.
- 💻 **Full coding agent training example:** Using only **6K training samples**, an end-to-end Qwen3.5-9B workflow improves SWE-bench Verified from **41.8% to 56.4%**, a gain of **14.6 percentage points**. We release the full pipeline, including data cleaning, reward-hacking prevention, and training scripts.

## ⚡ Installation

The following is an example installation on a CUDA 13.0 machine:

```bash
cd <this-repo>
uv sync
bash scripts/setup_verl.sh 0.8.0 cu130
```

See the [Installation Guide](https://microsoft.github.io/agent-lightning/stable/1-installation/) for details.


## ⚡ Architecture

<p align="center">
  <img src="docs/images/architecture.jpg" alt="Agent Lightning v1.0 architecture" width="800">
</p>

Agent Lightning v1.0 keeps the training architecture simple with three lightweight components:

- **Trainer:** Runs `verl` and vLLM, builds training samples, and updates the policy.
- **API Gateway:** Proxies model requests and captures training data.
- **Rollout Controller:** Runs agents locally or as Kubernetes Jobs.

The Trainer creates rollouts, the Controller launches agents, and the Gateway turns interactions into training data, while agents continue to run with their real harnesses.

## ⚡ Results

We evaluate Agent Lightning v1.0 across several practical training domains, including Search R1, LLM-in-Sandbox, and Coding Agent. Pure RL delivers substantial improvements across all three domains, as shown below.

<p align="center">
  <img src="docs/images/benchmark-comparison.jpg" alt="Agent Lightning v1.0 benchmark comparison" width="600">
</p>

## ⚡ Documentation

| Section | Content |
|---------|---------|
| [Installation](https://microsoft.github.io/agent-lightning/stable/1-installation/) | Base environment and `verl` GPU stack |
| [Quick Start](https://microsoft.github.io/agent-lightning/stable/2-quick-start/) | Local first run and end-to-end flow |
| [Basics](https://microsoft.github.io/agent-lightning/stable/3-basics/) | Components, rollouts, events, and trajectories |
| [Trainer Configuration](https://microsoft.github.io/agent-lightning/stable/4-trainer-configuration/) | `verl` integration and trace aggregation |
| [API Gateway Configuration](https://microsoft.github.io/agent-lightning/stable/5-api-gateway-configuration/) | Gateway and model proxy settings |
| [Controller Configuration](https://microsoft.github.io/agent-lightning/stable/6-controller-configuration/) | Local and Kubernetes runners |
| [Asynchronous Training](https://microsoft.github.io/agent-lightning/stable/7-asynchronous-training/) | Collocated async collection and pause/drain |

## ⚡ Examples

| Example | Description |
|---|---|
| [Calc-X](https://microsoft.github.io/agent-lightning/stable/8-example-calc-x/) | POC math reasoning example with AutoGen and MCP calculator tools, requiring only one GPU. |
| [GSM8K](https://microsoft.github.io/agent-lightning/stable/9-example-gsm8k/) | POC grade-school math reasoning example. |
| [ScienceWorld](https://microsoft.github.io/agent-lightning/stable/10-example-science-world/) | Interactive science tasks in a text-based environment. |
| [Search-R1](https://microsoft.github.io/agent-lightning/stable/11-example-search-r1/) | Multi-turn retrieval and reasoning agent. |
| [LLM-in-Sandbox](https://microsoft.github.io/agent-lightning/stable/12-example-llm-in-sandbox/) | General agent with computer and code execution tools. |
| [Coding Agent](https://microsoft.github.io/agent-lightning/stable/13-example-coding-agent/) | Coding agent trained with repository tests. |

## ⚡ Articles

- 12/17/2025 [Adopting the Trajectory Level Aggregation for Faster Training](https://agent-lightning.github.io/posts/trajectory_level_aggregation/) Agent-lightning blog.
- 11/4/2025 [Tuning ANY AI agent with Tinker ✕ Agent-lightning](https://medium.com/@yugez/tuning-any-ai-agent-with-tinker-agent-lightning-part-1-1d8c9a397f0e) Medium. See also [Part 2](https://medium.com/@yugez/tuning-any-ai-agent-with-tinker-agent-lightning-part-2-332c5437f0dc).
- 10/22/2025 [No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL](https://blog.vllm.ai/2025/10/22/agent-lightning.html) vLLM blog. See also [Zhihu writeup](https://zhuanlan.zhihu.com/p/1965067274642785725).
- 8/11/2025 [Training AI Agents to Write and Self-correct SQL with Reinforcement Learning](https://medium.com/@yugez/training-ai-agents-to-write-and-self-correct-sql-with-reinforcement-learning-571ed31281ad) Medium.
- 8/5/2025 [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680) arXiv paper.
- 7/26/2025 [We discovered an approach to train any AI agent with RL, with (almost) zero code changes.](https://www.reddit.com/r/LocalLLaMA/comments/1m9m670/we_discovered_an_approach_to_train_any_ai_agent/) Reddit.
- 6/6/2025 [Agent Lightning - Microsoft Research](https://www.microsoft.com/en-us/research/project/agent-lightning/) Project page.

## ⚡ Community Projects

- [DeepWerewolf](https://github.com/af-74413592/DeepWerewolf) — A case study of agent RL training for the Chinese Werewolf game built with AgentScope and Agent Lightning.
- [AgentFlow](https://agentflow.stanford.edu/) — A modular multi-agent framework that combines planner, executor, verifier, and generator agents with the Flow-GRPO algorithm to tackle long-horizon, sparse-reward tasks.
- [Youtu-Agent](https://github.com/TencentCloudADP/Youtu-agent) — Youtu-Agent lets you build and train your agent with ease. Built with [a modified branch](https://github.com/microsoft/agent-lightning/tree/contrib/youtu-agent-lightning) of Agent Lightning, Youtu-Agent has verified up to 128 GPUs RL training on maths/code and search capabilities with steady convergence. Also check [the recipe](https://github.com/TencentCloudADP/youtu-agent/tree/rl/agl) and their blog [*Stop Wrestling with Your Agent RL: How Youtu-Agent Achieved Stable, 128-GPU Scaling Without Breaking a Sweat*](https://spotted-coconut-df8.notion.site/Stop-Wrestling-with-Your-Agent-RL-How-Youtu-Agent-Achieved-Stable-128-GPU-Scaling-Without-Breaking-2ca5e8f089ba80539a98c582b65e0233).

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

This project welcomes contributions and suggestions. Start by reading the [Contributing Guide](docs/community/contributing.md) for recommended contribution points, environment setup, branching conventions, and pull request expectations. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## ⚡ Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## ⚡ Responsible AI

This project has been evaluated and certified to comply with the Microsoft Responsible AI Standard. The team will continue to monitor and maintain the repository, addressing any severe issues, including potential harms, if they arise.


## ⚡ License

Agent Lightning v1.0 is released under the [MIT License](LICENSE).
