from __future__ import annotations

import weakref
from typing import Any, Optional, TYPE_CHECKING

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
        self, train_dataset: Optional[Dataset[Any]] = None, validation_dataset: Optional[Dataset[Any]] = None
    ) -> None:
        """Subclasses should implement this method to implement the algorithm.

        Args:
            train_dataset: The dataset to train on. Not all algorithms require a training dataset.
            val_dataset: The dataset to validate on. Not all algorithms require a validation dataset.

        Returns:
            Algorithm should refrain from returning anything. It should just run the algorithm.
        """
        pass
