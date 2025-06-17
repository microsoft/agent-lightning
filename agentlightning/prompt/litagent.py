from __future__ import annotations

from typing import Any, List, Union, Optional, TYPE_CHECKING

from .types import NamedResources, Rollout, Task, TaskInput, Triplet

if TYPE_CHECKING:
    from .trainer import Trainer


class LitAgent:
    """Base class for the training and validation logic of an agent.

    Developers should subclass this class and implement the rollout methods
    to define the agent's behavior for a single task. The agent's logic
    is completely decoupled from the server communication and training
    infrastructure.
    """

    _trainer: Trainer | None = None

    def __init__(self, *, trained_agents: Optional[str] = None) -> None:  # FIXME: str | None won't work for cli
        """
        Initialize the LitAgent.

        Args:
            trained_agents: Optional string representing the trained agents.
                            This can be used to track which agents have been trained by this instance.
        """
        self.trained_agents = trained_agents

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer for this agent.

        Args:
            trainer: The Trainer instance that will handle training and validation.
        """
        self._trainer = trainer

    @property
    def trainer(self) -> Trainer:
        """
        Get the trainer for this agent.

        Returns:
            The Trainer instance associated with this agent.
        """
        if self._trainer is None:
            raise ValueError("Trainer has not been set for this agent.")
        return self._trainer

    def training_rollout(
        self, task: TaskInput, rollout_id: str, resources: NamedResources
    ) -> Union[float, List[Triplet], Rollout]:
        """Defines the agent's behavior for a single training task.

        This method should contain the logic for how the agent processes an
        input, uses the provided resources (like LLMs or prompts), and
        produces a result.

        Args:
            task: The task object received from the server, containing the
                  input data and metadata.
            rollout_id: A unique identifier for the rollout, used for tracking
                        and reporting purposes.
            resources: A dictionary of named resources (e.g., LLMs, prompt
                       templates) for the agent to use.

        Returns:
            The result of the rollout, which can be one of:
            - None. The tracing should be handled by the agent runner.
            - A float representing the final reward.
            - A list of `Triplet` objects for detailed, step-by-step feedback.
            - A complete `Rollout` object for full control over reporting.
        """
        raise NotImplementedError("Subclasses must implement the `training_rollout` method.")

    def validation_rollout(
        self, task: TaskInput, rollout_id: str, resources: NamedResources
    ) -> Union[float, List[Triplet], Rollout]:
        """Defines the agent's behavior for a single validation task.

        By default, this method redirects to `training_rollout`. Override it
        if the agent should behave differently during validation.

        Args:
            task: The task object received from the server, containing the
                  input data and metadata.
            rollout_id: A unique identifier for the validation rollout,
                        used for tracking and reporting purposes.
            resources: A dictionary of named resources for the agent to use.

        Returns:
            The result of the validation rollout. See `training_rollout` for
            possible return types.
        """
        return self.training_rollout(task, rollout_id, resources)

    async def training_rollout_async(
        self, task: TaskInput, rollout_id: str, resources: NamedResources
    ) -> Union[float, List[Triplet], Rollout]:
        """Asynchronous version of `training_rollout`.

        This method should be implemented by agents that perform asynchronous
        operations (e.g., non-blocking I/O, concurrent API calls).

        Args:
            task: The task object received from the server.
            rollout_id: A unique identifier for the training rollout,
                        used for tracking and reporting purposes.
            resources: A dictionary of named resources for the agent to use.

        Returns:
            The result of the asynchronous training rollout.
        """
        raise NotImplementedError("Async agents must implement the `training_rollout_async` method.")

    async def validation_rollout_async(
        self, task: TaskInput, rollout_id: str, resources: NamedResources
    ) -> Union[float, List[Triplet], Rollout]:
        """Asynchronous version of `validation_rollout`.

        By default, this method redirects to `training_rollout_async`.
        Override it for different asynchronous validation behavior.

        Args:
            task: The task object received from the server.
            rollout_id: A unique identifier for the validation rollout,
                        used for tracking and reporting purposes.
            resources: A dictionary of named resources for the agent to use.

        Returns:
            The result of the asynchronous validation rollout.
        """
        return await self.training_rollout_async(task, rollout_id, resources)
