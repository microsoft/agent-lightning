# Command Line Interface

<!-- TODO: This document should be auto-generated. -->

!!! warning

    This document is a work in progress and might not be updated with the latest changes.
    Try to use `agl -h` to get the latest help message.

!!! tip

    Agent-lightning also provides utilities to help you build your own CLI for [LitAgent][agentlightning.LitAgent] and [Trainer][agentlightning.Trainer]. See [Trainer](./trainer.md) for references.

## agl

```text
usage: agl [-h] {vllm,store,prometheus,agentops}

Agent Lightning CLI entry point.

Available subcommands:
  vllm        Run the vLLM CLI with Agent Lightning instrumentation.
  store       Run a LightningStore server.
  prometheus  Serve Prometheus metrics from the multiprocess registry.
  agentops    Start the AgentOps server manager.

positional arguments:
  {vllm,store,prometheus,agentops}
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

## agl prometheus

Expose the Prometheus multiprocess registry on a dedicated FastAPI server. This is useful when the main LightningStore service is under heavy load; exporters can scrape this auxiliary endpoint instead.

```text
usage: agl prometheus [-h] [--host HOST] [--port PORT] [--metrics-path METRICS_PATH] [--log-level {DEBUG,INFO,WARNING,ERROR}] [--access-log]

Serve Prometheus metrics outside the LightningStore server.

options:
  -h, --help            show this help message and exit
  --host HOST           Host to bind the metrics server to.
  --port PORT           Port to expose the Prometheus metrics on.
  --metrics-path METRICS_PATH
                        HTTP path used to expose metrics. Must start with '/' and not be the root path.
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Configure the logging level for the metrics server.
  --access-log          Enable uvicorn access logs. Disabled by default to reduce noise.
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
