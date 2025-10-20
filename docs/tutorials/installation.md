# Installation

Agent-lightning can be installed from PyPI or from resource.

## Install Stable Release

This will install the latest stable version of Agent-lightning from PyPI.

```bash
pip install --upgrade agentlightning
```

!!! tip

    If you are running Agent-lightning with VERL or running Agent-lightning's examples, you will also need to install [Algorithm-specific dependencies][algorithm-specific-installation] and [Example-specific dependencies][example-specific-installation]. See below for more details.

## Install Nightly Build

Agent-lightning will publish a build of new features on main branch every day. Install it from test PyPI.

```bash
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agentlightning
```

Nightly build is less stable than "stable release". So use it at your own risk.

[]{ #algorithm-specific-installation }

## Algorithm-specific Installation

### APO

To install [APO](../algorithm-zoo/apo.md), you need to install dependencies like [POML](https://github.com/microsoft/POML). Agent-lightning will install it for you with the following command.

```bash
pip install agentlightning[apo]
```

!!! warning

    [APO](../algorithm-zoo/apo.md) also relies on a compatible version of [OpenAI Python SDK](https://github.com/openai/openai-python) installed, preferably `>=2.0`.

### VERL

To install [VERL](../algorithm-zoo/verl.md), you shall have a compatible version of PyTorch, vLLM and VERL installed. You can but you are not advised to use the following command:

```bash
pip install agentlightning[verl]
```

!!! tip "More Robust Approach"

    The approach above can sometimes introduce dependency conflicts or missing dependencies with CUDA if you don't have a pre-installed PyTorch and CUDA toolkit. Use the following commands to install the dependencies manually:

    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    pip install flash-attn --no-build-isolation
    pip install vllm==0.10.2
    pip install verl==0.5.0
    ```

[]{ #example-specific-installation }

## Example-specific Installation

See [README]({{ src("examples") }}) of each example on how to install the dependencies for that example.

## Install from Source Code

You can choose to install Agent-lightning from source code if you are seeking to contribute to this project, or you want an isolated environment for running examples.

!!! note

    Starting from v0.2, we have migrated to [uv](https://docs.astral.sh/uv/) as the default dependency manager for Agent-lightning contributors. uv speeds up the legacy pip approach from minutes to seconds. It's also safer in managing dependency conflicts, pinning dependency versions and organizing dependencies into groups.

For minimal installation, you can use the following command. Note that this command relies on [uv](https://docs.astral.sh/uv/) to be installed beforehand.

```bash
git clone https://github.com/microsoft/agent-lightning
cd agent-lightning
uv sync --group dev
```

The `uv sync` command can also be used to install [Algorithm-specific dependencies][algorithm-specific-installation] and [Example-specific dependencies][example-specific-installation]. For full installation on a machine with no GPU accelerator, use:

```bash
uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-cpu \
    --group torch-stable \
    --group trl \
    --group agents \
    --no-default-groups
```

If you have a machine with a GPU accelerator, and the GPU is compatible with CUDA 12.8, you can use the following command to install the dependencies:

```bash
uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-gpu-stable \
    --group trl \
    --group agents \
    --no-default-groups
```

uv creates a virtual environment in the `.venv` directory. To use the environment you just created, you can use the following command:

```bash
# Option 1: Add uv run before every command you want to run
uv run python ...

# Option 2: Activate the environment
source .venv/bin/activate
python ...
```

!!! warning "Caution"

    Before contributing, also make sure to install the pre-commit hooks, which will save you a lot of time fixing unnecessary linting errors.

    ```bash
    uv run pre-commit install
    uv run pre-commit run --all-files --show-diff-on-failure --color=always
    ```
