# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to use an existing APO algorithm to tune the prompts."""

from typing import Tuple, cast

from openai import AsyncOpenAI
from room_selector import RoomSelectionTask, load_room_tasks, room_selector

from agentlightning import Trainer, configure_logger
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset


def load_train_val_dataset() -> Tuple[Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]]:
    dataset_full = load_room_tasks()
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    return cast(Dataset[RoomSelectionTask], dataset_train), cast(Dataset[RoomSelectionTask], dataset_val)


def main() -> None:
    openai_client = AsyncOpenAI()

    algo = APO[RoomSelectionTask](openai_client)
    trainer = Trainer(n_workers=1, algorithm=algo)
    dataset_train, dataset_val = load_train_val_dataset()
    trainer.fit_v2(agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val)


if __name__ == "__main__":
    configure_logger()

    main()
