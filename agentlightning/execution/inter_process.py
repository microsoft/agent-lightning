# Copyright (c) Microsoft. All rights reserved.

from .base import ExecutionStrategy


class InterProcessExecutionStrategy(ExecutionStrategy):

    alias: str = "ipc"

    # TODO: to be implemented
