# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
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
    "Resource",
    "LLM",
    "ProxyLLM",
    "PromptTemplate",
    "ResourceUnion",
    "NamedResources",
    "ResourcesUpdate",
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

logger = logging.getLogger(__name__)


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


class Resource(BaseModel):
    """
    Base class for all tunable resources.
    """

    resource_type: Any


class LLM(Resource):
    """
    Provide an LLM endpoint and model name as a resource.

    Attributes:
        endpoint (str): The URL of the LLM API endpoint.
        model (str): The identifier for the model to be used (e.g., 'gpt-4o').
        sampling_parameters (SamplingParameters): A dictionary of hyperparameters
            for model inference, such as temperature, top_p, etc.
    """

    resource_type: Literal["llm"] = "llm"
    endpoint: str
    model: str
    api_key: Optional[str] = None
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)

    def base_url(self, *args: Any, **kwargs: Any) -> str:
        """The base_url to put into openai.OpenAI.

        Users are encouraged to use `base_url` to get the LLM endpoint instead of accessing `endpoint` directly.
        """
        return self.endpoint


class ProxyLLM(LLM):
    """Proxy LLM resource that is tailored by `llm_proxy.LLMProxy`."""

    resource_type: Literal["proxy_llm"] = "proxy_llm"  # type: ignore
    _initialized: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Mark initialization as complete after Pydantic finishes setup."""
        super().model_post_init(__context)
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name: str) -> Any:
        """Override to emit a warning when endpoint is accessed directly."""
        # Check if we're accessing endpoint after initialization and not from base_url
        if name == "endpoint":
            try:
                initialized = object.__getattribute__(self, "_initialized")
            except AttributeError:
                initialized = False

            if initialized:
                # Check the call stack to see if we're being called from base_url
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_name = frame.f_back.f_code.co_name
                    if caller_name != "base_url":
                        logger.warning(
                            "Accessing 'endpoint' directly on ProxyLLM is discouraged. "
                            "Use 'base_url(rollout_id, attempt_id)' instead to get the properly formatted endpoint."
                        )
        return super().__getattribute__(name)

    def with_attempted_rollout(self, rollout: AttemptedRollout) -> LLM:
        """Bake the rollout and attempt id into the endpoint."""
        return LLM(
            endpoint=self.base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            model=self.model,
            sampling_parameters=self.sampling_parameters,
            api_key=self.api_key,
        )

    def base_url(self, rollout_id: Optional[str], attempt_id: Optional[str]) -> str:
        if rollout_id is None and attempt_id is None:
            return self.endpoint

        if not (isinstance(rollout_id, str) and isinstance(attempt_id, str)):
            raise ValueError("rollout_id and attempt_id must be strings or all be empty")

        prefix = self.endpoint
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        if prefix.endswith("/v1"):
            prefix = prefix[:-3]
            has_v1 = True
        else:
            has_v1 = False
        # Now the prefix should look like "http://localhost:11434"

        # Append the rollout and attempt id to the prefix
        prefix = prefix + f"/rollout/{rollout_id}/attempt/{attempt_id}"
        if has_v1:
            prefix = prefix + "/v1"
        return prefix


class PromptTemplate(Resource):
    """
    A prompt template as a resource.

    Attributes:
        template (str): The template string. The format depends on the engine.
        engine (Literal['jinja', 'f-string', 'poml']): The templating engine
            to use for rendering the prompt. I imagine users can use their own
            customized engines, but algos can only well operate on a subset of them.
    """

    resource_type: Literal["prompt_template"] = "prompt_template"
    template: str
    engine: Literal["jinja", "f-string", "poml"]


# Use discriminated union for proper deserialization
ResourceUnion = Annotated[Union[LLM, ProxyLLM, PromptTemplate], Field(discriminator="resource_type")]
NamedResources = Dict[str, ResourceUnion]
"""
A dictionary-like class to hold named resources.

Example:
    resources: NamedResources = {
        'main_llm': LLM(
            endpoint="http://localhost:8080",
            model="llama3",
            sampling_parameters={'temperature': 0.7, 'max_tokens': 100}
        ),
        'system_prompt': PromptTemplate(
            template="You are a helpful assistant.",
            engine='f-string'
        )
    }
"""


class ResourcesUpdate(BaseModel):
    """
    A resource update message to be sent from the server to clients.

    This message contains a dictionary of resources that clients should use
    for subsequent tasks. It is used to update the resources available to
    clients dynamically.
    """

    resources_id: str
    resources: NamedResources


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
