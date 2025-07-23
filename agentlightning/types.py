from typing import Any, Dict, List, Optional, Union, Literal, Annotated

from pydantic import BaseModel, Field, Discriminator
from opentelemetry.sdk.trace import ReadableSpan

__all__ = [
    "Triplet",
    "Rollout",
    "Task",
    "TaskInput",
    "TaskIfAny",
    "RolloutRawResult",
    "Resource",
    "LLM",
    "PromptTemplate",
    "ResourceUnion",
    "NamedResources",
    "ResourcesUpdate",
    "GenericResponse",
    "ParallelWorkerBase",
]


class Triplet(BaseModel):
    """A standard structure for a single turn in a trajectory."""

    prompt: Any
    response: Any
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Rollout(BaseModel):
    """The standard reporting object from client to server."""

    rollout_id: str

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


TaskInput = Any


class Task(BaseModel):
    """A task (rollout request) to be processed by the client agent."""

    rollout_id: str
    input: TaskInput

    mode: Optional[Literal["train", "val", "test"]] = None
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
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)


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
ResourceUnion = Annotated[Union[LLM, PromptTemplate], Field(discriminator="resource_type")]
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
