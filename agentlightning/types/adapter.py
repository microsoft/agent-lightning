# Copyright (c) Microsoft. All rights reserved.

"""Data formats used by adapters, usually the target format converted from trace spans."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
)

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import BaseModel, Field

from agentlightning.semconv import LinkPydanticModel

from .tracer import Attributes

T = TypeVar("T")


# General containers


class Tree(Generic[T]):
    """This is a generic tree data structure that can be used to represent the structure of a tree."""

    def __init__(self, item: T, children: MutableSequence[Tree[T]]) -> None:
        self.item = item
        self.children = children

    def traverse(self) -> Iterable[T]:
        yield self.item
        for child in self.children:
            yield from child.traverse()

    def count(self) -> int:
        return 1 + sum(child.count() for child in self.children)

    def __iter__(self) -> Iterator[T]:
        return iter(self.traverse())

    def __len__(self) -> int:
        return self.count()

    def add(self, child: Tree[T]) -> None:
        self.children.append(child)

    def prune(self, predicate: Callable[[T], bool]) -> Tree[T]:
        return Tree(self.item, [child.prune(predicate) for child in self.children if predicate(child.item)])

    def visualize(self, filename: str, item_to_str: Callable[[T], str]) -> None:
        """Render the tree with Graphviz for debugging purposes.

        Args:
            filename: Base filename for the generated `.png` diagram (without extension).

        !!! note

            The method requires the optional `graphviz` dependency to be available in the runtime
            environment.
        """
        import graphviz

        dot = graphviz.Digraph(comment="Tree")

        def visit(node: Tree[T]):
            dot.node(str(id(node)), item_to_str(node.item))  # type: ignore
            for child in node.children:
                visit(child)
                dot.edge(str(id(node)), str(id(child)))  # type: ignore

        visit(self)
        dot.render(filename, format="png", cleanup=True)  # type: ignore


# Annotation-related types


class Annotation(BaseModel):
    """An annotation is an approach to parse a span into some kind of structured attachments to another object.

    Note that a span can be parsed in multiple ways, and annotation is just one of them.
    """

    annotation_type: Literal["agent", "general", "message", "object", "exception", "operation"]
    """Type of the annotation."""

    span_id: str
    """Span ID of the annotation span. Not necessarily an [AGL_ANNOTATION][agentlightning.semconv.AGL_ANNOTATION] span."""

    links: Optional[Sequence[LinkPydanticModel]] = None
    """Links to other spans or objects."""


class AgentAnnotation(Annotation):
    """Parsed from [OTel Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)."""

    annotation_type = "agent"
    """Type of the annotation."""

    id: Optional[str] = None
    """The unique identifier of the GenAI agent."""

    name: Optional[str] = None
    """Human-readable name of the GenAI agent provided by the application."""

    description: Optional[str] = None
    """Free-form description of the GenAI agent provided by the application."""


class GeneralAnnotation(Annotation):
    """An annotation payload that is parsed from an [annotation][agentlightning.semconv.AGL_ANNOTATION] span."""

    annotation_type = "general"
    """Type of the annotation."""

    reward: Dict[str, float] = Field(default_factory=dict)
    """Reward dimensions and values."""

    primary_reward: Optional[float] = None
    """Primary reward value."""

    tag: Sequence[str] = Field(default_factory=list)
    """Tags for the annotation."""

    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    """Raw payload from the annotation."""


class MessageAnnotation(Annotation):
    """A log message that is parsed from a [message][agentlightning.semconv.AGL_MESSAGE] span."""

    annotation_type = "message"
    """Type of the annotation."""

    message: str
    """Message text."""


class ObjectAnnotation(Annotation):
    """An artifact that is parsed from a [object][agentlightning.semconv.AGL_OBJECT] span."""

    annotation_type = "object"
    """Type of the annotation."""

    object: Any
    """The object payload."""


class ExceptionAnnotation(Annotation):
    """An exception that is parsed from an [exception][agentlightning.semconv.AGL_EXCEPTION] span."""

    annotation_type = "exception"
    """Type of the annotation."""

    type: str
    """Type of the exception."""

    message: str
    """Message of the exception."""

    stacktrace: Optional[str] = None
    """Stacktrace of the exception."""


class OperationAnnotation(Annotation):
    """An operation that is parsed from an [operation][agentlightning.semconv.AGL_OPERATION] span."""

    annotation_type = "operation"
    """Type of the annotation."""

    name: str
    """Name of the operation."""

    input: Optional[Any] = None
    """Input of the operation."""

    output: Optional[Any] = None
    """Output of the operation."""


class ChatCompletionCall(BaseModel):
    """Corresponding to exactly one chat completion call.

    OpenAI chat completion request and response are used as standards here.
    Convert to other chat completion formats if needed.
    """

    request: CompletionCreateParams
    """OpenAI chat completion request parameters."""

    response: ChatCompletion
    """OpenAI chat completion response payload."""

    malformed_fields: Dict[str, Attributes]
    """Fields that are not supported by the adapter.

    Mapping from span names to a dict of malformed fields.
    """

    span_ids: Sequence[str]
    """Span IDs of the spans that contributed to this chat completion."""


class AnnotatedChatCompletionCall(ChatCompletionCall):
    """A chat completion call with annotations."""

    annotations: Sequence[Annotation]
    """Annotations for the chat completion call."""


# Algorithm-specific requirements


class TokenInput(BaseModel):
    """Token-based model input."""

    token_ids: Sequence[int]
    """Token IDs of the model input."""

    image_urls: Any
    """A list of image URLs. Could be pointers to local files or base64-encoded images."""


class TokenOutput(BaseModel):
    """Token-based model output."""

    token_ids: Sequence[int]
    """Token IDs of the model output."""


class TokenInputOutputTriplet(BaseModel):
    """A triplet of token IDs for the input and output, useful for reinforcement learning.

    This is not a stable interface and the fields here highly depend on RL implementations.
    """

    observation: TokenInput
    """Observation for the model input. Corresponding to prompt."""

    action: TokenOutput
    """Action, corresponding to completion result."""

    reward: Optional[float]
    """Reward of the model input."""

    done: bool
    """Whether it's the end of the trajectory."""

    raw_call: AnnotatedChatCompletionCall
    """Raw chat completion call."""


class AccumulatedTokenSequence(TokenInput):
    """A sequence of token IDs that are accumulated from multiple model calls.

    Output is implied in the token IDs.
    """

    response_mask: Sequence[int]
    """Mask for the response tokens. Must a sequence of 0s and 1s, with 1s for the completion tokens and 0s for the prompt tokens."""

    final_reward: Optional[float]
    """Single reward value for the entire sequence."""

    raw_calls: Sequence[AnnotatedChatCompletionCall]
    """Raw chat completion calls. The order of the calls must be the same as the order of the token IDs."""


class AccumulatedMessages(BaseModel):
    """A conversation that is accumulated from multiple model calls."""

    messages: Sequence[ChatCompletionMessageParam]
    """Messages of the conversation."""

    tools: Optional[Sequence[ChatCompletionFunctionToolParam]]
    """Tools provided for the conversation."""

    final_reward: Optional[float]
    """Single reward value for the entire conversation."""

    raw_calls: Sequence[AnnotatedChatCompletionCall]
    """Raw chat completion calls. The order of the calls must be the same as the order of the messages."""
