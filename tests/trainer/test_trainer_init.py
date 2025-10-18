# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os

import agentlightning as agl

# Initialize trainer with predefined tracer
algorithm = agl.Baseline()
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,
    tracer=agl.OtelTracer(),
)
# Runner is initialized to be the default runner: LitAgentRunner
assert isinstance(trainer.runner, agl.LitAgentRunner)
assert isinstance(trainer.runner.tracer, agl.OtelTracer)

# Use strategy alias "shm"
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=1,  # n_runners must be 1 here
    strategy="shm",
)
assert isinstance(trainer.strategy, agl.SharedMemoryExecutionStrategy)

# Use dict. Now n_runners can be >1 because algorithm is on the main thread
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,
    strategy={"type": "shm", "main_thread": "algorithm", "managed_store": False},
)
assert isinstance(trainer.strategy, agl.SharedMemoryExecutionStrategy)
assert trainer.strategy.main_thread == "algorithm"
assert trainer.strategy.managed_store is False

# n_runners is ignored in the trainer because strategy has been initialized with n_runners=4
strategy = agl.SharedMemoryExecutionStrategy(main_thread="algorithm", n_runners=4)
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,
    strategy=strategy,
)
assert trainer.strategy is strategy
assert trainer.strategy.n_runners == 4  # type: ignore

# By default, strategy is client-server, but you can also use a string alias to specify it again
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,
    strategy={
        # This line is optional
        "type": "cs",
        "server_port": 9999,
    },
)
assert isinstance(trainer.strategy, agl.ClientServerExecutionStrategy)
assert trainer.strategy.server_port == 9999

# Execution strategy supports using environment variables to override the values
os.environ["AGL_SERVER_PORT"] = "10000"
os.environ["AGL_CURRENT_ROLE"] = "algorithm"
os.environ["AGL_MANAGED_STORE"] = "0"
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,
    # This line is optional
    strategy="cs",
)
assert isinstance(trainer.strategy, agl.ClientServerExecutionStrategy)
assert trainer.strategy.server_port == 10000
assert trainer.strategy.role == "algorithm"
assert trainer.strategy.managed_store is False

# Use a similar approach to customize adapter
trainer = agl.Trainer(algorithm=algorithm, n_runners=8, adapter="agentlightning.adapter.TraceToMessages")
assert isinstance(trainer.adapter, agl.TraceToMessages)

# If it's a dict and type is not provided, it will use the default class
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=8,
    adapter={"agent_match": "plan_agent", "repair_hierarchy": False},
)
assert isinstance(trainer.adapter, agl.TracerTraceToTriplet)
assert trainer.adapter.agent_match == "plan_agent"
assert trainer.adapter.repair_hierarchy is False
