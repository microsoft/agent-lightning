# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, List, Literal, Optional

from agentlightning.types import Attempt, Dataset, Rollout, RolloutStatus, Span

from .base import Algorithm

logger = logging.getLogger(__name__)

__all__ = ["FastAlgorithm", "Baseline"]


class FastAlgorithm(Algorithm):
    """Base class for lightweight algorithms optimised for developer workflows.

    Fast algorithms prioritise short feedback loops so an agent developer can run
    small-scale experiments without waiting for long-running training jobs to
    finish.
    """


def _timestamp_to_iso_str(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


class Baseline(FastAlgorithm):
    """Reference implementation that streams the full dataset through the rollout queue.

    The baseline algorithm batches task submissions, waits for each rollout to
    finish, and logs every collected span and reward. It is primarily useful as
    a smoke test for the platform plumbing rather than a performant trainer.

    Args:
        n_epochs: Number of dataset passes to execute for both the train and val
            splits during developer experiments.
        train_split: Fraction of the concatenated dataset to treat as training
            data. Must be strictly between 0 and 1.
        polling_interval: Interval, in seconds, to poll the store for queue
            depth and rollout completion.
        max_queue_length: Number of rollouts allowed to wait in the queue before
            throttling additional submissions.
        span_verbosity: Level of detail to include when logging span metadata.

    Raises:
        ValueError: If `train_split` falls outside the `(0, 1)` interval.

    Examples:
        ```python
        from agentlightning.algorithm.fast import Baseline

        algorithm = Baseline(n_epochs=2, train_split=0.8, span_verbosity="key_values")
        trainer.fit(algorithm, train_dataset=my_train, val_dataset=my_val)
        ```
    """

    def __init__(
        self,
        *,
        n_epochs: int = 1,
        train_split: float = 0.5,
        polling_interval: float = 5.0,
        max_queue_length: int = 4,
        span_verbosity: Literal["keys", "key_values", "none"] = "keys",
    ) -> None:
        super().__init__()
        self.n_epochs = n_epochs
        self.train_split = train_split
        self.polling_interval = polling_interval
        self.max_queue_length = max_queue_length
        self.span_verbosity = span_verbosity
        if not (0.0 < self.train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1.")

        self._finished_rollout_count = 0

    def _span_to_string(self, rollout_id: str, attempt: Attempt, span: Span) -> str:
        """Format a span for logging based on the configured verbosity."""
        if self.span_verbosity == "none":
            return ""

        prefix_msg = f"[Rollout {rollout_id} | Attempt {attempt.attempt_id} | Span {span.span_id}] #{span.sequence_id} ({span.name}) "
        elapsed = f"{span.end_time - span.start_time:.2f}" if span.start_time and span.end_time else "unknown"

        msg = (
            prefix_msg
            + f"From {_timestamp_to_iso_str(span.start_time) if span.start_time else 'unknown'}, "
            + f"to {_timestamp_to_iso_str(span.end_time) if span.end_time else 'unknown'}, "
            + f"{elapsed} seconds. "
        )
        if self.span_verbosity == "key_values":
            msg += f"Attributes: {span.attributes}"
        else:
            msg += f"Attribute keys: {list(span.attributes.keys())}"
        return msg

    async def _handle_rollout_finish(self, rollout: Rollout) -> None:
        """Log attempt metadata and emit adapted traces when a rollout ends."""
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
                "[Rollout %s | Attempt %s] ID: %s. Status: %s. Worker: %s",
                rollout_id,
                attempt.sequence_id,
                attempt.attempt_id,
                attempt.status,
                attempt.worker_id,
            )
            spans = await store.query_spans(rollout_id=rollout_id)
            for span in spans:
                if self.span_verbosity != "none":
                    logger.info(self._span_to_string(rollout.rollout_id, attempt, span))

        # Attempts to adapt the spans using the adapter if provided
        try:
            adapter = self.get_adapter()
        except ValueError:
            logger.warning("No adapter set for MockAlgorithm. Skipping trace adaptation.")
            adapter = None
        if adapter is not None:
            spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
            transformed_data = adapter.adapt(spans)
            logger.info(f"[Rollout {rollout_id}] Adapted data: {transformed_data}")

    async def _enqueue_rollouts(
        self, dataset: Dataset[Any], train_indices: List[int], val_indices: List[int], resources_id: str
    ) -> None:
        """Submit rollouts while respecting the maximum queue length."""
        store = self.get_store()

        for index in train_indices + val_indices:
            queuing_rollouts = await store.query_rollouts(status_in=["queuing", "requeuing"])
            if len(queuing_rollouts) <= 1:
                # Only enqueue a new rollout when there is at most 1 rollout in the queue.
                sample = dataset[index]
                mode = "train" if index in train_indices else "val"
                rollout = await store.enqueue_rollout(input=sample, mode=mode, resources_id=resources_id)
                logger.info(f"[Rollout {rollout.rollout_id}] Enqueued in {mode} mode with sample: {sample}")
            await asyncio.sleep(self.polling_interval)

    async def _harvest_rollout_spans(self, rollout_id: str):
        """Poll rollout status updates until completion and log transitions."""
        store = self.get_store()
        last_status: Optional[RolloutStatus] = None

        while True:
            rollout = await store.get_rollout_by_id(rollout_id)
            if rollout is not None:
                if rollout.status in ["succeeded", "failed", "cancelled"]:
                    # Rollout is finished, log all the data.
                    await self._handle_rollout_finish(rollout)
                    # We are done here.
                    self._finished_rollout_count += 1
                    logger.info(f"Finished {self._finished_rollout_count} rollouts.")
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
    ) -> None:
        """Execute the baseline loop across the provided datasets."""
        train_dataset_length = len(train_dataset) if train_dataset is not None else 0
        val_dataset_length = len(val_dataset) if val_dataset is not None else 0
        if train_dataset_length == 0 and val_dataset_length == 0:
            logger.error(
                "MockAlgorithm requires at least one dataset. Provide train_dataset or val_dataset before running."
            )
            return

        concatenated_dataset = [train_dataset[i] for i in range(train_dataset_length) if train_dataset is not None] + [
            val_dataset[i] for i in range(val_dataset_length) if val_dataset is not None
        ]
        train_indices = list(range(0, train_dataset_length))
        val_indices = list(range(train_dataset_length, train_dataset_length + val_dataset_length))
        logger.debug(f"Train indices: {train_indices}")
        logger.debug(f"Val indices: {val_indices}")

        store = self.get_store()

        # Currently we only supports a single resource update at the start.
        initial_resources = self.get_initial_resources()
        if initial_resources is not None:
            resource_update = await store.update_resources("default", initial_resources)
            resources_id = resource_update.resources_id
            logger.info(f"Initial resources set: {initial_resources}")
        else:
            logger.warning("No initial resources provided. Skip initializing resources.")
            resources_id = None

        for epoch in range(self.n_epochs):
            harvest_tasks: List[asyncio.Task[None]] = []
            logger.info(f"Proceeding epoch {epoch + 1}/{self.n_epochs}.")
            for index in train_indices + val_indices:
                logger.info(
                    f"Processing index {index}. {len(train_indices)} train indices and {len(val_indices)} val indices in total."
                )
                while True:
                    queuing_rollouts = await store.query_rollouts(status_in=["queuing", "requeuing"])
                    if len(queuing_rollouts) <= self.max_queue_length:
                        # Only enqueue a new rollout when there is at most "max_queue_length" rollout in the queue.
                        sample = concatenated_dataset[index]
                        mode = "train" if index in train_indices else "val"
                        rollout = await store.enqueue_rollout(input=sample, mode=mode, resources_id=resources_id)
                        harvest_tasks.append(asyncio.create_task(self._harvest_rollout_spans(rollout.rollout_id)))
                        logger.info(f"Enqueued rollout {rollout.rollout_id} in {mode} mode with sample: {sample}")
                        break
                    else:
                        # Sleep a bit and try again later.
                        await asyncio.sleep(self.polling_interval)

            # Wait for all harvest tasks to complete
            logger.info(f"Waiting for {len(harvest_tasks)} harvest tasks to complete...")
            if len(harvest_tasks) > 0:
                await asyncio.gather(*harvest_tasks)
