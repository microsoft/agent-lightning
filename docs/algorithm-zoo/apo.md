# APO

!!! tip "Shortcut"

    You can use the shortcut `agl.APO(...)` to create an APO instance.

    ```python
    import agentlightning as agl

    agl.APO(...)
    ```

## Installation

```bash
pip install agentlightning[apo]
```

## Scope of Current Implementation

APO is currently scoped to optimize a single prompt template. Optimizing multiple prompt templates is not supported yet.

There is however no restriction on the number of variable placeholders in the prompt template (can range from zero to many). It's possible that invalid prompts are created during the optimization process. It is up to the agent developer to ensure that the prompt template is valid for the agent's task.

## Initial Prompt

APO expects the initial prompt to be provided in the `initial_resources` dictionary. This can be done in two approaches:

1. Pass to the [Trainer][agentlightning.Trainer] constructor:

```python
trainer = agl.Trainer(
    algorithm=agl.APO(...),
    initial_resources={"main_prompt": agl.PromptTemplate(template="You are a helpful assistant.", engine="f-string")},
)
```

2. Pass to the `[APO][agentlightning.algorithm.apo.APO].set_initial_resources()` method:

```python
algo = agl.APO(...)
algo.set_initial_resources(
    {"this_is_also_valid_key": agl.PromptTemplate(template="You are a helpful assistant.", engine="f-string")}
)
```

The resource key can be arbitrary, which is used to identify the prompt template in [class-based implementations](../tutorials/write-agents.md) when you have multiple resources. When the key changes, the agent developer needs to update the key in the `rollout` method.

## Tutorials Using APO

- [Train the First Agent with APO](../how-to/train-first-agent.md) - A step-by-step guide to training your first agent using APO.

## References

::: agentlightning.algorithm.apo
