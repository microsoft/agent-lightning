# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Optional,
    Union,
)

from agentlightning.adapter import TraceAdapter
from agentlightning.client import AgentLightningClient
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset, NamedResources

if TYPE_CHECKING:
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.trainer import Trainer


class Algorithm:
    """Algorithm is the strategy, or tuner to train the agent."""

    _trainer_ref: weakref.ReferenceType[Trainer] | None = None
    _llm_proxy_ref: weakref.ReferenceType["LLMProxy"] | None = None
    _store: LightningStore | None = None
    _initial_resources: NamedResources | None = None
    _adapter_ref: weakref.ReferenceType[TraceAdapter[Any]] | None = None

    def is_async(self) -> bool:
        """Return True if the algorithm is asynchronous."""
        return inspect.iscoroutinefunction(self.run)

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer for this algorithm.

        Args:
            trainer: The Trainer instance that will handle training and validation.
        """
        self._trainer_ref = weakref.ref(trainer)

    def get_trainer(self) -> Trainer:
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

    def set_llm_proxy(self, llm_proxy: LLMProxy | None) -> None:
        """
        Set the LLM proxy for this algorithm to reuse when available.

        Args:
            llm_proxy: The LLMProxy instance configured by the trainer, if any.
        """
        self._llm_proxy_ref = weakref.ref(llm_proxy) if llm_proxy is not None else None

    def get_llm_proxy(self) -> Optional[LLMProxy]:
        """
        Retrieve the configured LLM proxy instance, if one has been set.

        Returns:
            The active LLMProxy instance or None when not configured.
        """
        if self._llm_proxy_ref is None:
            return None

        llm_proxy = self._llm_proxy_ref()
        if llm_proxy is None:
            raise ValueError("LLM proxy reference is no longer valid (object has been garbage collected).")

        return llm_proxy

    def set_adapter(self, adapter: TraceAdapter[Any]) -> None:
        """
        Set the adapter for this algorithm to collect and convert traces.
        """
        self._adapter_ref = weakref.ref(adapter)

    def get_adapter(self) -> TraceAdapter[Any]:
        """
        Retrieve the adapter for this algorithm to communicate with the runners.
        """
        if self._adapter_ref is None:
            raise ValueError("Adapter has not been set for this algorithm.")
        adapter = self._adapter_ref()
        if adapter is None:
            raise ValueError("Adapter reference is no longer valid (object has been garbage collected).")
        return adapter

    def set_store(self, store: LightningStore) -> None:
        """
        Set the store for this algorithm to communicate with the runners.

        Store is set directly instead of using weakref because its copy is meant to be
        maintained throughout the algorithm's lifecycle.
        """
        self._store = store

    def get_store(self) -> LightningStore:
        """
        Retrieve the store for this algorithm to communicate with the runners.
        """
        if self._store is None:
            raise ValueError("Store has not been set for this algorithm.")
        return self._store

    def get_initial_resources(self) -> Optional[NamedResources]:
        """
        Get the initial resources for this algorithm.
        """
        return self._initial_resources

    def set_initial_resources(self, resources: NamedResources) -> None:
        """
        Set the initial resources for this algorithm.
        """
        self._initial_resources = resources

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> Union[None, Awaitable[None]]:
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

        Deprecated and will be removed in a future version.

        Returns:
            The AgentLightningClient instance associated with this algorithm.
        """
        raise NotImplementedError("Subclasses must implement get_client().")
