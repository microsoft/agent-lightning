# VERL

!!! tip "Shortcut"

    You can use the shortcut `agl.VERL(...)` to create a VERL instance.

    ```python
    import agentlightning as agl

    agl.VERL(...)
    ```

## Installation

```bash
pip install agentlightning[verl]
```

!!! warning

    To avoid various compatibility issues, follow the steps in the [installation guide](../tutorials/installation.md) to set up VERL and its dependencies. Installing VERL directly with `pip install agentlightning[verl]` can cause issues unless you already have a compatible version of PyTorch installed.

!!! note "Notes for Readers"

    [VERL][agentlightning.algorithm.verl.VERL] in this article refers to a wrapper, provided by Agent-lightning, of the [VERL framework](https://github.com/volcengine/verl). It's a subclass of [agentlightning.Algorithm][]. To differentiate it from the VERL framework, all references to the VERL framework shall use the term "VERL framework", and all references to the Agent-lightning wrapper shall be highlighted with a link.

## Resources

[VERL][agentlightning.algorithm.verl.VERL] expects no initial resources. The first LLM endpoint is directly deployed from the VERL configuration (`.actor_rollout_ref.model.path`). The resource key is always `main_llm`.

[VERL][agentlightning.algorithm.verl.VERL] currently does not support optimizing multiple [LLM][agentlightning.LLM]s together.

!!! note

    The resource type created by [VERL][agentlightning.algorithm.verl.VERL] is actually a [ProxyLLM][agentlightning.ProxyLLM], a subclass of the [LLM][agentlightning.LLM] type. This object contains a **URL template** provided by [VERL][agentlightning.algorithm.verl.VERL], with placeholders for rollout and attempt IDs. When a rollout begins on the agent side, the framework uses the current `rollout_id` and `attempt_id` to format this template, generating a final, unique endpoint URL. This URL points to [VERL][agentlightning.algorithm.verl.VERL]'s internal proxy, allowing it to intercept and log all traffic for that specific attempt, for tracing and load balancing purposes. For agents created with the `@rollout` decorator, this resolution of the template is handled automatically ("auto-stripped"). Class-based agents will need to manually resolve the [`ProxyLLM`][agentlightning.ProxyLLM] using the rollout context.

    ```python
    proxy_llm = resources["main_llm"]
    proxy_llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
    ```

## Customization

Internally, [VERL][agentlightning.algorithm.verl.VERL] decomposes each agent execution into prompt–response pairs via the [Adapter][agentlightning.Adapter] and associates them with their corresponding reward signals as [Triplet][agentlightning.Triplet] objects. The final scalar reward, derived from the last triplet in the trajectory, is propagated to all preceding triplets following the [identical assignment strategy](https://arxiv.org/abs/2508.03680). This ensures that each triplet receives an identical reward signal and can be independently optimized as a valid RLHF trajectory within the VERL framework.

At present, [VERL][agentlightning.algorithm.verl.VERL] does not expose fine-grained control over its reward propagation or credit assignment mechanisms. Users requiring customized reward shaping or trajectory decomposition are advised to clone and modify the [VERL][agentlightning.algorithm.verl.VERL] source implementation directly.

## Tutorials Using VERL

- [Train SQL Agent with RL](../how-to/train-sql-agent.md) - A practical example of training a SQL agent using VERL.

## References - Entrypoint

::: agentlightning.algorithm.verl

## References - Implementation

::: agentlightning.verl
