# Copyright (c) Microsoft. All rights reserved.

from vllm.entrypoints.cli.main import main

from agentlightning.instrumentation.vllm import instrument_vllm

if __name__ == "__main__":
    instrument_vllm()
    main()
