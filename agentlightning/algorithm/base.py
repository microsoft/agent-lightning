# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import weakref
from typing import Any, Optional, TYPE_CHECKING

from agentlightning.client import AgentLightningClient
from agentlightning.types import Dataset

if TYPE_CHECKING:
    from agentlightning.trainer import Trainer


class BaseAlgorithm:
    """Algorithm is the strategy, or tuner to train the agent."""

    _trainer_ref: weakref.ReferenceType[Trainer] | None = None

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer for this algorithm.

        Args:
            trainer: The Trainer instance that will handle training and validation.
        """
        self._trainer_ref = weakref.ref(trainer)

    @property
    def trainer(self) -> Trainer:
        """
        Get the trainer for this algorithm.

        Returns:
            The Trainer instance associated with this agent.
        """
        if self._trainer_ref is None:
            raise ValueError("Trainer has not been set for this agent.")
        trainer = self._trainer_ref()
        if trainer is None:
            raise ValueError("Trainer reference is no longer valid (object has been garbage collected).")
        return trainer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        validation_dataset: Optional[Dataset[Any]] = None,
        dev_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        """Subclasses should implement this method to implement the algorithm.

        Args:
            train_dataset: The dataset to train on. Not all algorithms require a training dataset.
            val_dataset: The dataset to validate on. Not all algorithms require a validation dataset.

        Returns:
            Algorithm should refrain from returning anything. It should just run the algorithm.
        """
        raise NotImplementedError("Subclasses must implement run().")

    def get_client(self) -> AgentLightningClient:
        """Get the client to communicate with the algorithm.

        If the algorithm does not require a server-client communication, it can also create a mock client
        that never communicates with itself.

        Returns:
            The AgentLightningClient instance associated with this algorithm.
        """
        raise NotImplementedError("Subclasses must implement get_client().")

    def fit(
        self,
        agent: Any,
        train_data: Optional[Dataset[Any]] = None,
        test_data: Optional[Dataset[Any]] = None,
        dev_data: Optional[Dataset[Any]] = None,
        trainer: Optional[Trainer] = None,
    ) -> None:
        """Fit the algorithm with the provided agent and datasets.

        Args:
            agent: The agent to train.
            train_data: The training dataset.
            test_data: The test dataset.
            dev_data: The development dataset.
            trainer: The trainer instance.
        """
        if trainer is not None:
            self.set_trainer(trainer)

        self.run(
            train_dataset=train_data,
            validation_dataset=test_data,
            dev_dataset=dev_data,
        )
