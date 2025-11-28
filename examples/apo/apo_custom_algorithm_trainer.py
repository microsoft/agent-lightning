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
from agentlightning.store import LightningStore

console = Console()


@algo
async def apo_algorithm_usable_in_trainer(*, store: LightningStore):
    """
    You need to wrap the apo_algorithm in an algo decorator to make it usable in trainer.

    This is equivalent to the following:

        apo_algorithm_usable_in_trainer = algo(apo_algorithm)
    """
    return await apo_algorithm(store=store)


if __name__ == "__main__":
    setup_logging()
    trainer = Trainer(n_workers=1, algorithm=apo_algorithm_usable_in_trainer)
    trainer.fit(apo_rollout)
