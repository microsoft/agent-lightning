# Algorithm Zoo

AgentLightning includes several popular and frequently requested algorithms in its built-in library, allowing agent developers to use them directly. These algorithms are designed to be compatible with most agent scenarios.

For customizing algorithms, see [Algorithm-side References](./algorithm.md).

## APO

!!! tip "Shortcut"

    You can use the shortcut `agl.APO(...)` to create an APO instance.

    ```python
    import agentlightning as agl

    agl.APO(...)
    ```

### Installation

```bash
pip install agentlightning[apo]
```

### Tutorials Using APO

TBD

### References

::: agentlightning.algorithm.apo

## VERL

!!! tip "Shortcut"

    You can use the shortcut `agl.VERL(...)` to create a VERL instance.

    ```python
    import agentlightning as agl

    agl.VERL(...)
    ```

### Installation

```bash
pip install agentlightning[verl]
```

!!! warning

    For best results, follow the steps in the [installation guide](../quickstart/installation.md) to set up VERL and its dependencies. Installing VERL directly with `pip install agentlightning[verl]` can cause issues unless you already have a compatible version of PyTorch installed.

### Tutorials Using VERL

TBD

### References - Entrypoint

::: agentlightning.algorithm.verl

### References - Implementation

::: agentlightning.verl
