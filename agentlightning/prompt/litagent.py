from __future__ import annotations

from typing import Any, List, Union

from .types import NamedResources, Rollout, Task, TaskInput, Triplet


class LitAgent:
    """Base class for the training and validation logic of an agent.

    Developers should subclass this class and implement the rollout methods
    to define the agent's behavior for a single task. The agent's logic
    is completely decoupled from the server communication and training
    infrastructure.
    """

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
