# Copyright (c) Microsoft. All rights reserved.

"""Put components in this file to make them available to the Trainer.

Currently only used for ExecutionStrategy.
"""

ExecutionStrategyRegistry = {
    "shm": "agentlightning.execution.shared_memory.SharedMemoryExecutionStrategy",
    # "ipc": "agentlightning.execution.inter_process.InterProcessExecutionStrategy",
    "cs": "agentlightning.execution.client_server.ClientServerExecutionStrategy",
}
