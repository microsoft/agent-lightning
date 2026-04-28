# Copyright (c) Microsoft. All rights reserved.

import random
from typing import Iterator, List, Sequence, TypeVar

from agentlightning.types import Dataset

T_task = TypeVar("T_task")


def batch_iter_over_dataset(dataset: Dataset[T_task], batch_size: int) -> Iterator[Sequence[T_task]]:
    """
    Create an infinite iterator that yields batches from the dataset.

    When batch_size >= dataset size, yields the entire shuffled dataset repeatedly.
    When batch_size < dataset size, yields batches of the specified size, reshuffling
    after each complete pass through the dataset.

    Args:
        dataset: The dataset to iterate over.
        batch_size: The desired batch size.

    Yields:
        Sequences of tasks from the dataset. Each task appears at most once per epoch.
    """
    if batch_size >= len(dataset):
        while True:
            dataset_copy = [dataset[i] for i in range(len(dataset))]
            random.shuffle(dataset_copy)
            yield dataset_copy

    else:
        current_batch: List[int] = []
        while True:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            for index in indices:
                if index in current_batch:
                    continue
                current_batch.append(index)
                if len(current_batch) == batch_size:
                    yield [dataset[index] for index in current_batch]
                    current_batch = []
