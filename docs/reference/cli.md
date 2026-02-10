# Command Line Interface

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
usage: agl store [-h] [--host HOST] [--port PORT] [--cors-origin CORS_ORIGINS] [--log-level {DEBUG,INFO,WARNING,ERROR}] [--tracker {prometheus,console} [{prometheus,console} ...]] [--n-workers N_WORKERS] [--backend {memory,mongo}]
                 [--mongo-uri MONGO_URI]

Run a LightningStore server

options:
  -h, --help            show this help message and exit
  --host HOST           Host to bind the server to
  --port PORT           Port to run the server on
  --cors-origin CORS_ORIGINS
                        Allowed CORS origin. Repeat for multiple origins. Use '*' to allow all origins.
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Configure the logging level for the store.
  --tracker {prometheus,console} [{prometheus,console} ...]
                        Enable metrics tracking. Repeat for multiple trackers.
  --n-workers N_WORKERS
                        Number of workers to run in the server. When it's greater than 1, the server will be run using `mp` launch mode. Only applicable for zero-copy stores such as MongoDB backend.
  --backend {memory,mongo}
                        Backend to use for the store.
  --mongo-uri MONGO_URI
                        MongoDB URI to use for the store. Applicable only if --backend is 'mongo'.
```

!!! tip

    After launching the store via CLI, you can tell the [`Trainer`][agentlightning.Trainer] to use the store by passing the store address to the trainer.

    ```python
    store_client = agl.LightningStoreClient("http://localhost:4747")
    trainer = agl.Trainer(store=store_client, ...)
    ```

    See [using external store][debug-with-external-store] for more details.

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
