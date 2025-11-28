# Installation Guide

This guide explains how to install **Agent-Lightning**. You can install it from **PyPI** (the Python Package Index) for general use or directly from the **source code** if you plan to contribute or need fine-grained control over dependencies.

!!! info "Platform and Hardware Requirements"
    Agent-Lightning is officially supported on **Linux distributions** (Ubuntu 22.04 or later is recommended).
    At the moment **macOS and Windows** (outside of WSL2) are not supported.

    The Python runtime must be **Python 3.10 or newer**. We recommend using the latest patch release of Python 3.10, 3.11, or 3.12 to pick up performance and security updates.

    A **GPU is optional**—you only need CUDA-capable hardware if you plan to fine-tune model weights or run GPU-accelerated workloads. CPU-only environments are fully supported for evaluation and inference.

## Installing from PyPI

The easiest way to get started is by installing Agent-Lightning directly from PyPI. This ensures you get the latest **stable release** of the package, tested for compatibility and reliability.

### Install the Stable Release

Run the following command in your terminal:

```bash
pip install --upgrade agentlightning
```

This installs or upgrades Agent-Lightning to the newest stable version.

!!! tip

    If you intend to use **Agent-Lightning** with [**VERL**](../algorithm-zoo/verl.md) or run any of its **example scripts**, you’ll need to install some additional dependencies.
    See the sections on [Algorithm-specific installation](#algorithm-specific-installation) and [Example-specific installation](#example-specific-installation) for details.

### Install the Nightly Build (Latest Features)

Agent-Lightning also publishes **nightly builds**, which contain the latest experimental features and improvements from the main branch. These are available via **Test PyPI**.

```bash
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agentlightning
```

!!! warning
    The nightly builds are cutting-edge but may include unstable or untested changes.
    Use them **at your own risk**, especially in production environments.

## Algorithm-specific Installation

Agent-Lightning supports multiple learning algorithms. Some of them like [APO](../algorithm-zoo/apo.md) or [VERL](../algorithm-zoo/verl.md) require extra dependencies. You can install them automatically using **optional extras** or manually if you prefer finer control.

### Installing APO

[APO](../algorithm-zoo/apo.md) is an algorithm module that depends on libraries such as [POML](https://github.com/microsoft/POML).
You can install Agent-Lightning with APO support by running:

```bash
pip install agentlightning[apo]
```

!!! warning
    APO also depends on the [OpenAI Python SDK](https://github.com/openai/openai-python), version **2.0 or newer**.
    Ensure your SDK version is up to date to avoid compatibility issues.

### Installing VERL

[VERL](../algorithm-zoo/verl.md) integrates with libraries like **PyTorch**, **vLLM**, and **VERL framework**.
Although you *can* install all dependencies automatically, we recommend doing it manually to avoid version conflicts.

```bash
pip install agentlightning[verl]
```

!!! tip "Recommended Manual Setup (More Stable)"
    Automated installation may cause issues if you don’t have a compatible **PyTorch** or **CUDA** version preinstalled.
    For a more stable setup, install dependencies step-by-step:

    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    pip install flash-attn --no-build-isolation
    pip install vllm==0.10.2
    pip install verl==0.5.0
    ```

    This approach ensures compatibility with CUDA 12.8 and minimizes dependency conflicts.

## Example-specific Installation

Each example in the `examples/` directory may have its own additional dependencies.
Please refer to the **README** file of each example for detailed setup instructions:

[See Example READMEs]({{ src("examples") }}).

## Installing from Source (for Developers and Contributors)

If you plan to contribute to Agent-Lightning or prefer to work with the latest development code, install it directly from the **source repository**.

### Why Install from Source?

* You want to **modify or contribute** to the project.
* You prefer an **isolated development environment**.
* You want to test unreleased features or fix bugs locally.

### Using `uv` for Dependency Management

Starting with version **0.2**, Agent-Lightning uses [`uv`](https://docs.astral.sh/uv/) as its **default dependency manager**.

`uv` is a fast and safe alternative to `pip` that:

* Installs packages **in seconds** (instead of minutes),
* Prevents **dependency conflicts**,
* Supports **grouped dependencies** for optional features.

Before proceeding, make sure `uv` is installed.

### Minimal Developer Installation

```bash
git clone https://github.com/microsoft/agent-lightning
cd agent-lightning
uv sync --group dev
```

This command sets up a clean development environment with only the essential dependencies.

### Installing All Extras (CPU or GPU)

`uv sync` can also handle algorithm-specific and example-specific dependencies in one step.

For a CPU-only machine:

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

For a GPU-equipped machine that is CUDA 12.8 compatible:

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

Read more about Agent-lightning managed dependency groups [here]({{ src("pyproject.toml") }}).

### Building the Dashboard

The Agent-Lightning dashboard is built using [Vite](https://vite.dev/). To build the dashboard, run the following command:

```bash
cd dashboard
npm ci
npm run build
```

Some HTML and JavaScript assets will be generated in the `agentlightning/dashboard` directory.

### Activating Your Environment

After syncing dependencies, `uv` automatically creates a virtual environment inside the `.venv/` directory.

You can use it in two ways:

```bash
# Option 1: Prefix commands with uv run
uv run python your_script.py

# Option 2: Activate the virtual environment
source .venv/bin/activate
python your_script.py
```

!!! warning "Before Contributing"

    Agent-Lightning enforces code style and linting rules via **pre-commit hooks**.
    Installing them early prevents many avoidable formatting issues.

    ```bash
    uv run pre-commit install
    uv run pre-commit run --all-files --show-diff-on-failure --color=always
    ```
