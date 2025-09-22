# Copyright (c) Microsoft. All rights reserved.


class ExecutionStrategy:
    """
    Execution strategy determines the placements of and communications between roles.
    
    """




class SharedMemoryExecutionStrategy(ExecutionStrategy):
    alias: str = "shm"

    def __init__(self, n_rollout_workers: int = 1, main_thread: str = "rollout"):
        self.n_rollout_workers = n_rollout_workers
        self.main_thread = main_thread


class InterProcessExecutionStrategy(ExecutionStrategy):
    alias: str = "ipc"


class RayExecutionStrategy(ExecutionStrategy):


class ClientServerExecutionStrategy(ExecutionStrategy):
    alias: str = "cs"

    def __init__(self, role: str, ):
        # role can be either server or client
