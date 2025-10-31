# Algorithm Zoo

AgentLightning includes several popular and frequently requested algorithms in its built-in library, allowing agent developers to use them directly. These algorithms are designed to be compatible with most agent scenarios.

For customizing algorithms, see [Algorithm-side References](../reference/algorithm.md).

| Algorithm | Optimizing Resources | Description |
| --------- | ------------------- | ----------- |
| [APO](./apo.md) | `{<initial_prompt_key>: [PromptTemplate][agentlightning.PromptTemplate]}` | Automatic Prompt Optimization (APO) algorithm using textual gradients and beam search. |
| [VERL](./verl.md) | `{"main_llm": [LLM][agentlightning.LLM]}` | Reinforcement Learning with [VERL framework](https://github.com/volcengine/verl). |
