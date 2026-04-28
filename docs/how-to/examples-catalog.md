# Examples Catalog

!!! tip "Want to Contribute?"

    We welcome contributions to the examples catalog! Please refer to the [Contributing](../community/contributing.md) guide for more details.

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } __APO room selector__

    ---

    Prompt-optimize a room-booking agent with the built-in APO algorithm, then contrast it with the write-your-own algorithm and debugging workflows in the tutorials. Pairs well with the [Train the First Agent how-to]({{ src("docs/how-to/train-first-agent.md") }}) and the [Write the First Algorithm guide]({{ src("docs/how-to/write-first-algorithm.md") }}).

    [:octicons-repo-24: Browse source]({{ src("examples/apo") }})

-   :material-cloud-sync:{ .lg .middle } __Azure OpenAI SFT__

    ---

    Run a supervised fine-tuning loop against Azure OpenAI: roll out the capital-lookup agent, turn traces into JSONL, launch fine-tunes, and redeploy the resulting checkpoints through Azure CLI.

    [:octicons-repo-24: Browse source]({{ src("examples/azure") }})

-   :material-calculator:{ .lg .middle } __Calc-X VERL math__

    ---

    VERL-based reinforcement learning setup for a math-reasoning agent that uses AutoGen plus an MCP calculator tool to solve Calc-X problems end to end.

    [:octicons-repo-24: Browse source]({{ src("examples/calc_x") }})

-   :material-code-braces:{ .lg .middle } __Claude Code SWE-bench__

    ---

    Instrumented driver that runs Anthropic's Claude Code workflow on SWE-bench instances while streaming traces through Agent-lightning—supports hosted vLLM, official Anthropic, or any OpenAI-compatible backend and emits datasets for downstream tuning.

    [:octicons-repo-24: Browse source]({{ src("examples/claude_code") }})

-   :material-view-grid:{ .lg .middle } __Minimal building blocks__

    ---

    Bite-sized scripts that isolate Agent-lightning primitives (e.g., LightningStore usage, LLM proxying, minimal vLLM host) so you can study each part before composing larger workflows.

    [:octicons-repo-24: Browse source]({{ src("examples/minimal") }})

-   :material-book-open-page-variant:{ .lg .middle } __RAG (MuSiQue)__

    ---

    Retrieval-Augmented Generation pipeline that preps a Wikipedia retriever via MCP and trains a MuSiQue QA agent with GRPO. Documented for historical reference (verified on Agent-lightning v0.1.x).

    [:octicons-repo-24: Browse source]({{ src("examples/rag") }})

-   :material-magnify:{ .lg .middle } __Search-R1 RL__

    ---

    Reproduction of the Search-R1 workflow that prepares its own retrieval backend, runs the rollout script, and coordinates GRPO-style training without extra orchestration layers (last validated on v0.1.x).

    [:octicons-repo-24: Browse source]({{ src("examples/search_r1") }})

-   :material-database:{ .lg .middle } __Spider SQL agent__

    ---

    LangGraph-powered text-to-SQL workflow for the Spider benchmark, combining LangChain tooling with Agent-lightning rollouts; follow along with the [how-to for training SQL agents]({{ src("docs/how-to/train-sql-agent.md") }}).

    [:octicons-repo-24: Browse source]({{ src("examples/spider") }})

-   :material-thought-bubble:{ .lg .middle } __Tinker integration__

    ---

    Adapter package ([`agl_tinker`]({{ src("examples/tinker/agl_tinker") }})) with Tinker plus sample CrewAI/OpenAI agents that feed Agent-lightning traces into Tinker’s reinforcement-learning backend for both toy and 20-Questions-style workflows.

    [:octicons-repo-24: Browse source]({{ src("examples/tinker") }})

-   :material-fast-forward:{ .lg .middle } __Unsloth SFT__

    ---

    Supervised fine-tuning loop that ranks math-agent rollouts, fine-tunes with Unsloth’s 4-bit LoRA stack, and mirrors the [Fine-tune with Unsloth recipe]({{ src("docs/how-to/unsloth-sft.md") }}).

    [:octicons-repo-24: Browse source]({{ src("examples/unsloth") }})

</div>
