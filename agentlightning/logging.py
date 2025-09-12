from __future__ import annotations

import json
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np

from .types import Hook

if TYPE_CHECKING:
    from .types import Task, Rollout
    from .tracer import BaseTracer
    from .runner import AgentRunner


def configure_logger(level: int = logging.INFO, name: str = "agentlightning") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()  # clear existing handlers

    # log to stdout
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging
    return logger


logger = logging.getLogger(__name__)


class LightningLogger(Hook):
    """Agent-lightning logger that supports tracing events and metrics throughout the training."""

    def log_event(self, event: str, data: dict):
        """
        Log an event with its associated data when something happens.
        """

    def log_metric(self, metric: str, value: float, step: Optional[int] = None):
        """
        Log a metric with its value and an optional step.
        """

    def log_message(self, level: int, message: str):
        """
        Log a message at a specific logging level.
        """

    def on_rollout_end(self, task: Task, rollout: Rollout, runner: AgentRunner, tracer: BaseTracer):
        """
        By default, each logger automatically logs the rollout event at the end of each rollout.
        """
        self.log_event("rollout", {"task": task.model_dump(), "rollout": rollout.model_dump()})


class ConsoleLogger(LightningLogger):
    """A simple logger that logs messages to the console using Python's logging module."""

    def __init__(self, level: int = logging.INFO):
        self.logger = configure_logger(level, name="agentlightning.ConsoleLogger")
        self.default_level = level
        self.worker_id: Optional[int] = None

    def init_worker(self, worker_id: int):
        super().init_worker(worker_id)
        self.worker_id = worker_id

    def teardown_worker(self, worker_id: int):
        super().teardown_worker(worker_id)
        self.worker_id = None

    def log_event(self, event: str, data: dict):
        data_str = str(data)
        if len(data_str) > 512:
            data_str = f"{data_str[:512]}... (truncated)"
        message = f"Event: {event}, Data: {data_str}"
        self.log_message(self.default_level, message)

    def log_metric(self, metric: str, value: float, step: Optional[int] = None):
        step_str = f" at step {step}" if step is not None else ""
        message = f"Metric: {metric} = {value}{step_str}"
        self.log_message(self.default_level, message)

    def log_message(self, level: int, message: str):
        if level >= self.default_level:
            if self.worker_id is not None:
                message = f"(Worker-{self.worker_id}) {message}"
            else:
                message = f"(Main) {message}"
            self.logger.log(level, message)
        # else skip logging if below default level


class WandbLogger(LightningLogger):

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        *,
        flush_every_n_events: int = 128,
        aggregate_every_n_metrics: int = 128,
    ):
        import wandb
        from wandb.sdk.wandb_run import Run

        self.wandb_run: Optional[Run] = None
        self.wandb_run_id: Optional[str] = None

        self.project = project
        self.entity = entity
        self.name = name or wandb.util.generate_id()
        self.config = config or {}

        self.event_table: Optional[wandb.Table] = None
        self.flush_every_n_events = flush_every_n_events
        self.aggregate_every_n_metrics = aggregate_every_n_metrics

        self.metrics_buffer: dict[str, list[float]] = {}

    def init_worker(self, worker_id: int):
        import wandb

        super().init_worker(worker_id)
        self.wandb_run = wandb.init(
            project=self.project,
            entity=self.entity,
            group=self.name,
            job_type=f"worker_{worker_id}",
            config=self.config,
        )
        logger.info(f"Wandb run initialized: {self.name} (Worker {worker_id})")
        if self.wandb_run is None:
            raise RuntimeError("Failed to initialize Wandb run.")
        self.wandb_run_id = self.wandb_run.id

    def teardown_worker(self, worker_id: int):
        import wandb

        super().teardown_worker(worker_id)

        for metric in self.metrics_buffer:
            if len(self.metrics_buffer[metric]) > 0:
                self._log_aggregated_metrics(metric)

        if len(self.event_table.data) > 0:
            logger.info(f"Flushing {len(self.event_table.data)} events to Wandb before finishing...")
            wandb.log({"client/events": self.event_table})
            self.event_table = None

        wandb.finish(exit_code=0)

    def teardown(self):
        import wandb

        super().teardown()
        if self.wandb_run is not None:
            wandb.finish(exit_code=0)
            self.wandb_run = None
            self.wandb_run_id = None

    def log_event(self, event: str, data: dict):
        import wandb

        if self.event_table is None:
            self.event_table = wandb.Table(columns=["event", "data"])

        try:
            data_str = json.dumps(data)  # Ensure data is JSON serializable
        except (TypeError, ValueError):
            data_str = str(data)
        self.event_table.add_data(event, data_str)

        if len(self.event_table.data) % self.flush_every_n_events == 0:
            logger.info(f"Flushing {len(self.event_table.data)} events to Wandb...")
            wandb.log({"client/events": self.event_table})

    def log_metric(self, metric: str, value: float, step: Optional[int] = None):
        import wandb

        if step is not None:
            wandb.log({"client_metric/" + metric: value}, step=step)
        else:
            wandb.log({"client_metric/" + metric: value})

        if metric not in self.metrics_buffer:
            self.metrics_buffer[metric] = []
        self.metrics_buffer[metric].append(value)
        if len(self.metrics_buffer[metric]) >= self.aggregate_every_n_metrics:
            self._log_aggregated_metrics(metric, step)
            self.metrics_buffer[metric] = []

    def log_message(self, level: int, message: str):
        pass  # Wandb handles logging internally, so we don't need to implement this

    def _log_aggregated_metrics(self, metric, step: Optional[int] = None):
        import wandb

        arr = np.array(self.metrics_buffer[metric])
        aggregated_value = {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "std": float(np.std(arr)),
            "count": int((~np.isnan(arr)).sum()),
        }
        for key, value in aggregated_value.items():
            if value is not None:
                if step is not None:
                    wandb.log({"client_agg/" + metric + "/" + key: value}, step=step)
                else:
                    wandb.log({"client_agg/" + metric + "/" + key: value})
