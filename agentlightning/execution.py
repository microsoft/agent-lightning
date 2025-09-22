# Copyright (c) Microsoft. All rights reserved.


class ExecutionStrategy:
    """
    Execution strategy determines the placements of and communications between roles.
    
    """

    def fork(self) -> None:
        ...

    def terminate(self) -> None:
        ...



class SharedMemoryExecutionStrategy(ExecutionStrategy):
    alias: str = "shm"

    def __init__(self, n_rollout_workers: int = 1, main_thread: str = "rollout"):
        self.n_rollout_workers = n_rollout_workers
        self.main_thread = main_thread

    def fork(self) -> None:
        ...


class InterProcessExecutionStrategy(ExecutionStrategy):
    alias: str = "ipc"


class ClientServerExecutionStrategy(ExecutionStrategy):
    alias: str = "cs"

    def __init__(self, role: str, )
