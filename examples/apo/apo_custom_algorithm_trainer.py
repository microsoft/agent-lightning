# Copyright (c) Microsoft. All rights reserved.

"""This sample code shows how to integrate a custom algorithm into trainer,
so that you can run it with one command:

```bash
python apo_custom_algorithm_trainer.py
```

This is equivalent to the following three commands in parallel:

```bash
agl store
python apo_custom_algorithm.py algo
python apo_custom_algorithm.py runner
```
"""

from apo_custom_algorithm import apo_algorithm, apo_rollout
from rich.console import Console

from agentlightning import Trainer, setup_logging
from agentlightning.algorithm import algo
from agentlightning.types import AlgorithmContext

console = Console()


@algo
async def apo_algorithm_usable_in_trainer(context: AlgorithmContext):
    """
    You need to wrap the apo_algorithm in an algo decorator to make it usable in trainer.

    This is equivalent to the following:

        async def wrapper(context: AlgorithmContext):
            return await apo_algorithm(store=context.store)

        apo_algorithm_usable_in_trainer = algo(wrapper)
    """
    return await apo_algorithm(store=context.store)


if __name__ == "__main__":
    setup_logging()
    trainer = Trainer(n_runners=1, algorithm=apo_algorithm_usable_in_trainer)
    trainer.fit(apo_rollout)
