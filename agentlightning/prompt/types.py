from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


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
    trace: Optional[List[Dict[str, Any]]] = None  # For serialized OpenTelemetry data
    logs: Optional[List[str]] = None

    # A bucket for any other relevant information
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskMetadata(BaseModel):
    """Metadata for a task, used to provide additional context."""

    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None

    class Config:
        extra = "allow"


class Task(BaseModel):
    """A task (rollout request) to be processed by the client agent."""

    rollout_id: str
    input: Any
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)


class TaskIfAny(BaseModel):
    is_available: bool
    task: Optional[Task] = None


class Resource(BaseModel):
    """
    Base class for all tunable resources.
    """


class LLM(Resource):
    """
    Provide an LLM endpoint and model name as a resource.

    Attributes:
        endpoint (str): The URL of the LLM API endpoint.
        model (str): The identifier for the model to be used (e.g., 'gpt-4o').
        sampling_parameters (SamplingParameters): A dictionary of hyperparameters
            for model inference, such as temperature, top_p, etc.
    """

    endpoint: str
    model: str
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)


class PromptTemplate(Resource):
    """
    A prompt template as a resource.

    Attributes:
        template (str): The template string. The format depends on the engine.
        engine (Literal['jinja', 'f-string', 'poml']): The templating engine
            to use for rendering the prompt.
    """

    template: str
    engine: Literal["jinja", "f-string", "poml"]


NamedResources = Dict[str, Resource]
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
