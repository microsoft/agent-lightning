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

## Resources

[VERL][agentlightning.algorithm.verl.VERL] expects no initial resources. The first LLM endpoint is directly deployed from the VERL configuration (`.actor_rollout_ref.model.path`). The resource key is always `main_llm`.

[VERL][agentlightning.algorithm.verl.VERL] currently does not support optimizing multiple [LLM][agentlightning.LLM]s together.

!!! note

    The resource type created by VERL is actually a [ProxyLLM][agentlightning.ProxyLLM], a subclass of the [LLM][agentlightning.LLM] type. This object contains a **URL template** provided by VERL, with placeholders for rollout and attempt IDs. When a rollout begins on the agent side, the framework uses the current `rollout_id` and `attempt_id` to format this template, generating a final, unique endpoint URL. This URL points to VERL's internal proxy, allowing it to intercept and log all traffic for that specific attempt, for tracing and load balancing purposes. For agents created with the `@rollout` decorator, this resolution of the template is handled automatically ("auto-stripped"). Class-based agents will need to manually resolve the `ProxyLLM` using the rollout context.

    ```python
    proxy_llm = resources["main_llm"]
    proxy_llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
    ```

## Customization

Internally, [VERL][agentlightning.algorithm.verl.VERL] extracts all prompt-response pairs with [Adapter][agentlightning.Adapter], and find their sibling reward signals as [Triplet][agentlightning.Triplet]s. The final reward is extracted from the last triplet's reward. Due to the [identical assignment strategy](https://arxiv.org/abs/2508.03680) being used, the final reward is always copied to all the triplets, and each triplet is considered as an RLHF trajectory to be optimized within VERL.


More advanced customization of VERL is not supported at this point. We recommend copying the source code from VERL and modifying it as needed to suit your requirements.

## Tutorials Using VERL

- [Train SQL Agent with RL](../how-to/train-sql-agent.md) - A practical example of training a SQL agent using VERL.

## References - Entrypoint

::: agentlightning.algorithm.verl

## References - Implementation

::: agentlightning.verl
