# Command Line Interface

<!-- TODO: This document should be auto-generated. -->

!!! warning

    This document is a work in progress and might not be updated with the latest changes.
    Try to use `agl -h` to get the latest help message.

!!! tip

    Agent-lightning also provides utilities to help you build your own CLI for [LitAgent][agentlightning.LitAgent] and [Trainer][agentlightning.Trainer]. See [Trainer](./trainer.md) for references.

## agl

```text
usage: agl [-h] {vllm,store,agentops}

Agent Lightning CLI entry point.

Available subcommands:
  vllm      Run the vLLM CLI with Agent Lightning instrumentation.
  store     Run a LightningStore server.
  agentops  Start the AgentOps server manager.

positional arguments:
  {vllm,store,agentops}
                        Subcommand to run.

options:
  -h, --help            show this help message and exit
```

## agl vllm

Agent-lightning's instrumented vLLM CLI.

```text
usage: agl vllm [-h] [-v] {chat,complete,serve,bench,collect-env,run-batch} ...

vLLM CLI

positional arguments:
  {chat,complete,serve,bench,collect-env,run-batch}
    chat                Generate chat completions via the running API server.
    complete            Generate text completions based on the given prompt via the running API server.
    collect-env         Start collecting environment information.
    run-batch           Run batch prompts and write results to file.

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

For full list:            vllm [subcommand] --help=all
For a section:            vllm [subcommand] --help=ModelConfig    (case-insensitive)
For a flag:               vllm [subcommand] --help=max-model-len  (_ or - accepted)
Documentation:            https://docs.vllm.ai
```

## agl store

Agent-lightning's LightningStore CLI. Use it to start an independent LightningStore server.

Currently the store data are stored in memory and will be lost when the server is stopped.

```text
usage: agl store [-h] [--port PORT]

Run a LightningStore server

options:
  -h, --help   show this help message and exit
  --port PORT  Port to run the server on
```

## agl agentops

Start a mock AgentOps server to bypass the online service of AgentOps.

```text
usage: agl agentops [-h] [--daemon] [--port PORT]

Start AgentOps server

options:
  -h, --help   show this help message and exit
  --daemon     Run server as a daemon
  --port PORT  Port to run the server on
```
