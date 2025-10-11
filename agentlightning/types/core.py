# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field, model_validator

from .tracer import Span

if TYPE_CHECKING:
    from agentlightning.litagent import LitAgent
    from agentlightning.runner.base import BaseRunner
    from agentlightning.tracer.base import BaseTracer

__all__ = [
    "Triplet",
    "Rollout",
    "Task",
    "TaskInput",
    "TaskIfAny",
    "RolloutRawResult",
    "RolloutRawResultV2",
    "RolloutMode",
    "GenericResponse",
    "ParallelWorkerBase",
    "Dataset",
    "AttemptStatus",
    "RolloutStatus",
    "RolloutConfig",
    "RolloutV2",
    "Attempt",
    "AttemptedRollout",
    "Hook",
]

T_co = TypeVar("T_co", covariant=True)


class Triplet(BaseModel):
    """A standard structure for a single turn in a trajectory."""

    prompt: Any
    response: Any
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Rollout(BaseModel):
    """The standard reporting object from client to server."""

    rollout_id: str

    # Echoing the input task
    task: Optional[Task] = None

    # Primary, high-level feedback
    final_reward: Optional[float] = None

    # Structured, sequential feedback for RL-style optimization
    triplets: Optional[List[Triplet]] = None

    # Optional, rich-context data for deep analysis
    trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of spans that conform to the OpenTelemetry JSON format. "
        "Users of the opentelemetry-sdk can generate this by calling "
        "json.loads(readable_span.to_json()).",
    )
    logs: Optional[List[str]] = None

    # A bucket for any other relevant information
    metadata: Dict[str, Any] = Field(default_factory=dict)


RolloutStatus = Literal[
    "queuing",  # initial status
    "preparing",  # after the trace is claimed
    "running",  # after receiving the first trace
    "failed",  # crashed
    "succeeded",  # status OK
    "cancelled",  # cancelled by user (or watchdog)
    "requeuing",  # retrying
]

AttemptStatus = Literal[
    # A status is essentially a process.
    # It should not have scheduling/management statuses like "queuing" or "cancelled".
    "preparing",
    "running",
    "failed",
    "succeeded",
    "unresponsive",  # the worker has not reported results for a while
    "timeout",  # the worker has been emitting new logs, but have been working on the task for too long
]

RolloutMode = Literal["train", "val", "test"]


class Attempt(BaseModel):
    """An attempt to execute a rollout. A rollout can have multiple attempts if retries are needed."""

    rollout_id: str  # the rollout this attempt belongs to
    attempt_id: str  # the universal id for current attempt
    sequence_id: int  # the sequence number of the attempt, starting from 1
    start_time: float  # time when the attempt has started
    end_time: Optional[float] = None  # time when the attempt has ended

    status: AttemptStatus = "preparing"
    # The rollout worker which is executing this attempt
    worker_id: Optional[str] = None

    last_heartbeat_time: Optional[float] = None  # last time when the worker has reported progress

    # A bucket for any other relevant information
    metadata: Optional[Dict[str, Any]] = None


class RolloutConfig(BaseModel):
    """Configurations for rollout execution."""

    timeout_seconds: Optional[float] = None  # none indicates no timeout
    unresponsive_seconds: Optional[float] = None  # none indicates no unresponsive timeout
    max_attempts: int = Field(default=1, ge=1)  # including the first attempt
    retry_condition: List[AttemptStatus] = Field(
        default_factory=cast(Callable[[], List[AttemptStatus]], list)
    )  # list of statuses that should trigger a retry


class RolloutV2(BaseModel):
    rollout_id: str

    # Inputs
    input: TaskInput

    # Time to track the lifecycle of the rollout
    start_time: float
    end_time: Optional[float] = None

    mode: Optional[RolloutMode] = None
    resources_id: Optional[str] = None

    # Overall scheduling/running information
    status: RolloutStatus = "queuing"

    config: RolloutConfig = Field(default_factory=RolloutConfig)

    # A bucket for any other relevant information
    metadata: Optional[Dict[str, Any]] = None


class AttemptedRollout(RolloutV2):
    """A rollout along with its active attempt."""

    attempt: Attempt

    @model_validator(mode="after")
    def check_consistency(self) -> AttemptedRollout:
        if self.attempt.rollout_id != self.rollout_id:
            raise ValueError("Inconsistent rollout_id between Rollout and Attempt")
        return self


TaskInput = Any


class Task(BaseModel):
    """A task (rollout request) to be processed by the client agent."""

    rollout_id: str
    input: TaskInput

    mode: Optional[RolloutMode] = None
    resources_id: Optional[str] = None

    # Optional fields for tracking task lifecycle
    create_time: Optional[float] = None
    last_claim_time: Optional[float] = None
    num_claims: Optional[int] = None

    # Allow additional metadata fields
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskIfAny(BaseModel):
    is_available: bool
    task: Optional[Task] = None


RolloutRawResult = Union[None, float, List[Triplet], List[Dict[str, Any]], List[ReadableSpan], Rollout]

RolloutRawResultV2 = Union[
    None,  # nothing (relies on tracer)
    float,  # only final reward
    List[ReadableSpan],  # constructed OTEL spans by user
    List[Span],  # constructed Span objects by user
]


class GenericResponse(BaseModel):
    """
    A generic response message that can be used for various purposes.
    """

    status: str = "success"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ParallelWorkerBase:
    """Base class for objects that can be parallelized across multiple worker processes.

    This class defines the standard lifecycle for parallel processing:

    Main Process:
        1. init() - Initialize the object in the main process
        2. spawn workers and call init_worker() in each worker
        3. run() - Execute the main workload in parallel across workers
        4. teardown_worker() - Clean up resources in each worker
        5. teardown() - Final cleanup in the main process

    Subclasses should implement the run() method and optionally override
    the lifecycle methods for custom initialization and cleanup behavior.
    """

    def __init__(self) -> None:
        """Initialize the base class. This method can be overridden by subclasses."""
        self.worker_id: Optional[int] = None

    def init(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        self.worker_id = worker_id

    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        pass

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        pass


class Dataset(Protocol, Generic[T_co]):
    """The general interface for a dataset.

    It's currently implemented as a protocol, having a similar interface to torch.utils.data.Dataset.
    You don't have to inherit from this class; you can use a simple list if you want to.
    """

    def __getitem__(self, index: int) -> T_co: ...

    def __len__(self) -> int: ...


class Hook(ParallelWorkerBase):
    """Base class for defining hooks in the agent runner's lifecycle."""

    async def on_trace_start(
        self, *, agent: LitAgent[Any], runner: BaseRunner[Any], tracer: BaseTracer, rollout: RolloutV2
    ) -> None:
        """Hook called immediately after the tracer enters the trace context but before the rollout begins.

        Args:
            agent: The :class:`LitAgent` instance associated with the runner.
            runner: The :class:`BaseRunner` managing the rollout.
            tracer: The :class:`BaseTracer` instance associated with the runner.
            rollout: The :class:`RolloutV2` object that will be processed.

        Subclasses can override this method to implement custom logic such as logging,
        metric collection, or resource setup. By default, this is a no-op.
        """

    async def on_trace_end(
        self, *, agent: LitAgent[Any], runner: BaseRunner[Any], tracer: BaseTracer, rollout: RolloutV2
    ) -> None:
        """Hook called immediately after the rollout completes but before the tracer exits the trace context.

        Args:
            agent: The :class:`LitAgent` instance associated with the runner.
            runner: The :class:`BaseRunner` managing the rollout.
            tracer: The :class:`BaseTracer` instance associated with the runner.
            rollout: The :class:`RolloutV2` object that has been processed.

        Subclasses can override this method to implement custom logic such as logging,
        metric collection, or resource cleanup. By default, this is a no-op.
        """

    async def on_rollout_start(self, *, agent: LitAgent[Any], runner: BaseRunner[Any], rollout: RolloutV2) -> None:
        """Hook called immediately before a rollout *attempt* begins.

        Args:
            agent: The :class:`LitAgent` instance associated with the runner.
            runner: The :class:`BaseRunner` managing the rollout.
            rollout: The :class:`RolloutV2` object that will be processed.

        Subclasses can override this method to implement custom logic such as
        logging, metric collection, or resource setup. By default, this is a
        no-op.
        """

    async def on_rollout_end(
        self,
        *,
        agent: LitAgent[Any],
        runner: BaseRunner[Any],
        rollout: RolloutV2,
        spans: Union[List[ReadableSpan], List[Span]],
    ) -> None:
        """Hook called after a rollout *attempt* completes.

        Args:
            agent: The :class:`LitAgent` instance associated with the runner.
            runner: The :class:`BaseRunner` managing the rollout.
            rollout: The :class:`RolloutV2` object that has been processed.
            spans: The spans that have been added to the store.

        Subclasses can override this method for cleanup or additional
        logging. By default, this is a no-op.
        """
