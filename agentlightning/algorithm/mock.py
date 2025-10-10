# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional

from agentlightning.llm_proxy import ModelConfig
from agentlightning.types import Dataset, NamedResources, RolloutStatus, RolloutV2

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


def _timestamp_to_iso_str(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


class MockAlgorithm(BaseAlgorithm):
    """A dummy implementation of algorithm interface that puts all dataset into the queue, and waits for all rollouts to complete.

    Logs all collected spans and rewards.

    Args:
        resources: Optional initial resources to set in the store.
        model_list: Optional list of models to load into the llm proxy.
            If both model_list and llm_proxy is provided, llm_proxy will be launched.
        n_epochs: Number of epochs to run through the dev dataset. Default is 1.
        train_split: Fraction of dev dataset to use for training vs validation. Must be between 0 and 1. Default is 0.5.
        polling_interval: Time interval (in seconds) to poll the store for queue length and for completed rollouts. Default is 5.0 seconds.
        max_queue_length: Maximum number of rollouts to keep in the queue at any time. Default is 1.
    """

    def __init__(
        self,
        *,
        resources: Optional[NamedResources] = None,
        model_list: Optional[List[ModelConfig]] = None,
        n_epochs: int = 1,
        train_split: float = 0.5,
        polling_interval: float = 5.0,
        max_queue_length: int = 1,
    ) -> None:
        super().__init__()
        self.resources = resources
        self.n_epochs = n_epochs
        self.train_split = train_split
        self.polling_interval = polling_interval
        self.max_queue_length = max_queue_length
        if not (0.0 < self.train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1.")

    async def _handle_rollout_finish(self, rollout: RolloutV2) -> None:
        store = self.get_store()

        rollout_id = rollout.rollout_id
        rollout_end_time = rollout.end_time or asyncio.get_event_loop().time()
        logger.info(
            f"[Rollout {rollout_id}] Finished with status {rollout.status} in {rollout_end_time - rollout.start_time:.2f} seconds."
        )

        # Logs all the attempts and their corresponding spans
        attempts = await store.query_attempts(rollout_id)
        for attempt in attempts:
            logger.info(
                f"[Rollout {rollout_id} | Attempt {attempt.sequence_id}] ID: {attempt.attempt_id}. Status: {attempt.status}. Worker: {attempt.worker_id}"
            )
            spans = await store.query_spans(rollout_id=rollout_id)
            for span in spans:
                prefix_msg = f"[Rollout {rollout_id} | Attempt {attempt.attempt_id} | Span {span.span_id}] #{span.sequence_id} ({span.name}) "
                logger.info(
                    prefix_msg
                    + f"From {_timestamp_to_iso_str(span.start_time) if span.start_time else 'unknown'}, "
                    + f"to {_timestamp_to_iso_str(span.end_time) if span.end_time else 'unknown'}, "
                    + f"{span.end_time - span.start_time if span.start_time and span.end_time else 'unknown'} seconds."
                )
                logger.info(prefix_msg + f"Attributes: {span.attributes}")

        # Attempts to adapt the spans using the adapter if provided
        try:
            adapter = self.get_adapter()
            spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
            transformed_data = await adapter.adapt(spans)
            logger.info(f"[Rollout {rollout_id}] Adapted data: {transformed_data}")
        except ValueError:
            logger.warning("No adapter set for MockAlgorithm. Skipping trace adaptation.")

    async def _enqueue_rollouts(
        self, dataset: Dataset[Any], train_indices: List[int], val_indices: List[int], resources_id: str
    ) -> None:
        store = self.get_store()

        for index in train_indices + val_indices:
            queuing_rollouts = await store.query_rollouts(status=["queuing", "requeuing"])
            if len(queuing_rollouts) <= 1:
                # Only enqueue a new rollout when there is at most 1 rollout in the queue.
                sample = dataset[index]
                mode = "train" if index in train_indices else "val"
                rollout = await store.enqueue_rollout(input=sample, mode=mode, resources_id=resources_id)
                logger.info(f"[Rollout {rollout.rollout_id}] Enqueued in {mode} mode with sample: {sample}")
            await asyncio.sleep(self.polling_interval)

    async def _harvest_rollout_spans(self, rollout_id: str):
        store = self.get_store()
        last_status: Optional[RolloutStatus] = None

        while True:
            rollout = await store.get_rollout_by_id(rollout_id)
            if rollout is not None:
                if rollout.status in ["succeeded", "failed", "cancelled"]:
                    # Rollout is finished, log all the data.
                    await self._handle_rollout_finish(rollout)
                    # We are done here.
                    break

                if last_status != rollout.status:
                    if last_status is not None:
                        logger.info(f"[Rollout {rollout_id}] Status changed to {rollout.status}.")
                    else:
                        logger.info(f"[Rollout {rollout_id}] Status is initialized to {rollout.status}.")
                    last_status = rollout.status

                else:
                    logger.debug(f"[Rollout {rollout_id}] Status is still {rollout.status}.")

            await asyncio.sleep(self.polling_interval)

    async def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
        dev_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        if dev_dataset is None or len(dev_dataset) == 0:
            logger.error("DummyAlgorithm requires a dev_dataset to run. No dev_dataset is provided. Exiting.")
            return

        # Split the dev dataset into train and validation sets
        split_idx = int(len(dev_dataset) * self.train_split)
        train_indices = list(range(0, split_idx))
        val_indices = list(range(split_idx, len(dev_dataset)))
        if len(train_indices) == 0:
            logger.warning("Train dataset is empty after split. All data will be used for validation.")
        if len(val_indices) == 0:
            logger.warning("Validation dataset is empty after split. All data will be used for training.")

        store = self.get_store()

        # Currently we only supports a single resource update at the start.
        if self.resources is not None:
            resource_update = await store.update_resources("default", self.resources)
            logger.info(f"Initial resources set: {self.resources}")
        else:
            resource_update = await store.update_resources("default", {})
            logger.info("No initial resources provided. Using empty resources.")

        for epoch in range(self.n_epochs):
            harvest_tasks: List[asyncio.Task[None]] = []
            logger.info(f"Proceeding epoch {epoch + 1}/{self.n_epochs}.")
            for index in train_indices + val_indices:
                queuing_rollouts = await store.query_rollouts(status=["queuing", "requeuing"])
                if len(queuing_rollouts) <= 1:
                    # Only enqueue a new rollout when there is at most 1 rollout in the queue.
                    sample = dev_dataset[index]
                    mode = "train" if index in train_indices else "val"
                    rollout = await store.enqueue_rollout(
                        input=sample, mode=mode, resources_id=resource_update.resources_id
                    )
                    harvest_tasks.append(asyncio.create_task(self._harvest_rollout_spans(rollout.rollout_id)))
                    logger.info(f"Enqueued rollout {rollout.rollout_id} in {mode} mode with sample: {sample}")
                await asyncio.sleep(self.polling_interval)

            # Wait for all harvest tasks to complete
            if len(harvest_tasks) > 0:
                await asyncio.gather(*harvest_tasks)
