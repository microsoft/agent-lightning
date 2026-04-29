# Copyright (c) Microsoft. All rights reserved.

"""Data formats used by adapters, usually the target format converted from trace spans."""

from __future__ import annotations

import logging
import weakref
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
    CompletionCreateParams,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agentlightning.semconv import LinkPydanticModel, RewardPydanticModel

from .tracer import Attributes, Span

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseAdaptingSequenceItem(Protocol):

    def with_container(self, container: BaseAdaptingSequence[Self]) -> Self:
        """Create a new item with the same properties but different container."""
        raise NotImplementedError()


T_co = TypeVar("T_co", covariant=True, bound=BaseAdaptingSequenceItem)
V_co = TypeVar("V_co", covariant=True, bound=BaseAdaptingSequenceItem)


class BaseAdaptingSequence(Sequence[T_co], Generic[T_co]):
    """Interface that makes adapter easier to work with sequences."""

    @overload
    def __getitem__(self, index: int) -> T_co: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T_co]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T_co, Sequence[T_co]]:
        return self.get(index)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.traverse())

    def __len__(self) -> int:
        return self.size()

    def get(self, index: Union[int, slice]) -> Union[T_co, Sequence[T_co]]:
        """Get the index-th item in the sequence."""
        raise NotImplementedError()

    def map(self, func: Callable[[T_co], V_co]) -> BaseAdaptingSequence[V_co]:
        """Map a function over all items in the sequence."""
        raise NotImplementedError()

    def retain(self, predicate: Callable[[T_co], bool]) -> BaseAdaptingSequence[T_co]:
        """Filter items in the sequence by a predicate (true for items to be kept).

        Depending on the implementation, the returned sequence may contain more or less items than a standard filter.
        """
        raise NotImplementedError()

    def prune(self, predicate: Callable[[T_co], bool]) -> BaseAdaptingSequence[T_co]:
        """Prune items in the sequence by a predicate (true for items to be pruned).

        Depending on the implementation, the returned sequence may contain more or less items than a standard prune.
        """
        raise NotImplementedError()

    def size(self) -> int:
        """Get the size of the sequence."""
        raise NotImplementedError()

    def traverse(self) -> Iterable[T_co]:
        """Traverse all items in the sequence."""
        raise NotImplementedError()


# General containers


class Tree(BaseAdaptingSequence[T_co], Generic[T_co]):
    """This is a generic tree data structure that can be used to represent the structure of a tree.

    This data structure is immutable.
    """

    def __init__(self, item: T_co, children: Sequence[Tree[T_co]]) -> None:
        self._children = children
        self._parent: Optional[weakref.ReferenceType[Tree[T_co]]] = None
        for child in self._children:
            child._parent = weakref.ref(self)  # type: ignore
        self._item = item.with_container(self)

    @property
    def item(self) -> T_co:
        return self._item

    @property
    def children(self) -> Sequence[Tree[T_co]]:
        return self._children

    @property
    def parent(self) -> Optional[Tree[T_co]]:
        return self._parent() if self._parent is not None else None

    def traverse(self) -> Iterable[T_co]:
        yield self._item
        for child in self._children:
            yield from child.traverse()

    def size(self) -> int:
        return 1 + sum(child.size() for child in self._children)

    def get(self, index: Union[int, slice]) -> Union[T_co, Sequence[T_co]]:
        """Get the index-th item in the tree (O(n) time complexity).

        I think this is not efficient, but it's seldomly used.
        """
        return list(self.traverse())[index]

    def map(self, func: Callable[[T_co], V_co]) -> Tree[V_co]:
        """Map a function over all items in the tree."""
        return Tree(func(self._item), [child.map(func) for child in self._children])

    def _retain_subtree(self, predicate: Callable[[T_co], bool]) -> Optional[Tree[T_co]]:
        if predicate(self._item):
            # If the current node satisfies the predicate, retain the subtree
            return self

        subtrees = [child._retain_subtree(predicate) for child in self._children]
        if all(subtree is None for subtree in subtrees):
            # no subtrees satisfy the predicate, remove the current node
            return None

        return Tree(self._item, [subtree for subtree in subtrees if subtree is not None])

    def retain(self, predicate: Callable[[T_co], bool]) -> Tree[T_co]:
        """Prune the tree by retaining subtrees with root nodes that satisfy the predicate.

        The root node is always retained.
        """
        return self._retain_subtree(predicate) or Tree(self._item, [])

    def prune(self, predicate: Callable[[T_co], bool]) -> Tree[T_co]:
        """Prune the tree by removing nodes that satisfy the predicate.

        The root node is always retained.
        """
        return Tree(self._item, [child.prune(predicate) for child in self._children if not predicate(child._item)])

    def visualize(self, filename: str, item_to_str: Callable[[T_co], str]) -> None:
        """Render the tree with Graphviz for debugging purposes.

        Args:
            filename: Base filename for the generated `.png` diagram (without extension).

        !!! note

            The method requires the optional `graphviz` dependency to be available in the runtime
            environment.
        """
        import graphviz

        dot = graphviz.Digraph(comment="Tree")

        def visit(node: Tree[T_co]):
            dot.node(str(id(node)), item_to_str(node.item))  # type: ignore
            for child in node._children:
                visit(child)
                dot.edge(str(id(node)), str(id(child)))  # type: ignore

        visit(self)
        dot.render(filename, format="png", cleanup=True)  # type: ignore


class AdaptingSequence(BaseAdaptingSequence[T_co], Generic[T_co]):
    """A simple list implementation of AdaptingSequence."""

    def __init__(self, items: Sequence[T_co]) -> None:
        # Set container on items if they are AdaptingSpan instances
        self._items = [item.with_container(self) for item in items]

    def get(self, index: Union[int, slice]) -> Union[T_co, Sequence[T_co]]:
        return self._items[index]

    def traverse(self) -> Iterable[T_co]:
        return iter(self._items)

    def size(self) -> int:
        return len(self._items)

    def map(self, func: Callable[[T_co], V_co]) -> AdaptingSequence[V_co]:
        return AdaptingSequence([func(item) for item in self._items])

    def retain(self, predicate: Callable[[T_co], bool]) -> AdaptingSequence[T_co]:
        return AdaptingSequence([item for item in self._items if predicate(item)])

    def prune(self, predicate: Callable[[T_co], bool]) -> AdaptingSequence[T_co]:
        return AdaptingSequence([item for item in self._items if not predicate(item)])


class AdaptingSpan(Span):
    """A span that has been adapted to a different format.

    This class extends the base [`Span`][agentlightning.Span] class to represent spans that have
    been converted to a different format by an adapter.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any
    """The data in the adapted format. Could be annotations, calls, or other structured data."""

    container: Optional[BaseAdaptingSequence[AdaptingSpan]] = None
    """An optional container that holds multiple adapted data items."""

    def with_data(self, data: Any, override: Literal["silent", "warning", "forbidden"] = "warning") -> AdaptingSpan:
        """Create a new [`AdaptingSpan`][agentlightning.AdaptingSpan] with the same properties but different adapted data.

        Args:
            data: The new adapted data.

        Returns:
            An instance of [`AdaptingSpan`][agentlightning.AdaptingSpan] with the same properties as
            the current span but with the provided adapted data.
        """
        if self.data is not None:
            if override == "forbidden":
                raise ValueError(
                    "Overwriting existing data on AdaptingSpan is forbidden. "
                    f"Current data: {self.data}, New data: {data}"
                )
            elif override == "warning":
                logger.warning(
                    "Found annotation on an adapting span with existing data; overwriting the data. "
                    f"Current data: {self.data}, New data: {data}"
                )
        return self.model_copy(update={"data": data})

    def with_container(self, container: BaseAdaptingSequence[AdaptingSpan]) -> AdaptingSpan:
        """Create a new [`AdaptingSpan`][agentlightning.AdaptingSpan] with the same properties but different container."""
        return self.model_copy(update={"container": container})

    @classmethod
    def from_span(cls, span: Span, data: Any) -> AdaptingSpan:
        """Create an [`AdaptingSpan`][agentlightning.AdaptingSpan] from a base [`Span`][agentlightning.Span].

        Args:
            span: The base span to convert.
            data: The data in the adapted format.

        Returns:
            An instance of [`AdaptingSpan`][agentlightning.AdaptingSpan] with the same properties as
            the input span and the provided adapted data.
        """
        if isinstance(span, AdaptingSpan):
            return span.model_copy(update={"data": data})
        else:
            return AdaptingSpan.model_validate({**span.model_dump(), "data": data})

    def children(self) -> Sequence[AdaptingSpan]:
        """Get the child spans as [`AdaptingSpan`][agentlightning.AdaptingSpan] instances.

        Only applicable when the container is a [`Tree`][agentlightning.Tree].
        """
        if self.container is None or not isinstance(self.container, Tree):
            raise ValueError("AdaptingSpan.children() is only applicable when container is non-empty and a Tree.")
        return [child.item for child in self.container.children]

    def siblings(self) -> Sequence[AdaptingSpan]:
        """Get the sibling spans as [`AdaptingSpan`][agentlightning.AdaptingSpan] instances.

        Only applicable when the container is a [`Tree`][agentlightning.Tree].
        """
        if self.container is None or not isinstance(self.container, Tree):
            raise ValueError("AdaptingSpan.siblings() is only applicable when container is non-empty and a Tree.")
        parent_tree = self.container.parent
        if parent_tree is None:
            return []
        return [child.item for child in parent_tree.children if child is not self.container]

    def parent_span(self) -> Optional[AdaptingSpan]:
        """Get the parent span if available.

        Only applicable when the container is a [`Tree`][agentlightning.Tree].
        """
        if self.container is None or not isinstance(self.container, Tree):
            raise ValueError("AdaptingSpan.parent() is only applicable when container is non-empty and a Tree.")
        parent_tree = self.container.parent
        return parent_tree.item if parent_tree is not None else None


# Annotation-related types

AnnotationType = Literal["agent", "general", "message", "object", "exception", "operation"]


class Annotation(BaseModel):
    """An annotation is an approach to parse a span into some kind of structured attachments to another object.

    Not necessarily an [AGL_ANNOTATION][agentlightning.semconv.AGL_ANNOTATION] span.

    Note that a span can be parsed in multiple ways, and annotation is just one of them.
    """

    annotation_type: AnnotationType
    """Type of the annotation."""

    links: Sequence[LinkPydanticModel] = Field(default_factory=list)  # type: ignore
    """Links to other spans or objects."""


class AgentAnnotation(Annotation):
    """Parsed from [OTel Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)."""

    annotation_type: AnnotationType = "agent"
    """Type of the annotation."""

    id: Optional[str] = None
    """The unique identifier of the GenAI agent."""

    name: Optional[str] = None
    """Human-readable name of the GenAI agent provided by the application."""

    description: Optional[str] = None
    """Free-form description of the GenAI agent provided by the application."""


class GeneralAnnotation(Annotation):
    """An annotation payload that is parsed from an [annotation][agentlightning.semconv.AGL_ANNOTATION] span."""

    annotation_type: AnnotationType = "general"
    """Type of the annotation."""

    rewards: Sequence[RewardPydanticModel] = Field(default_factory=list)  # type: ignore
    """Reward dimensions and values."""

    primary_reward: Optional[float] = None
    """Primary reward value."""

    tags: Sequence[str] = Field(default_factory=list)
    """Tags for the annotation."""

    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    """Raw payload from the annotation."""


class MessageAnnotation(Annotation):
    """A log message that is parsed from a [message][agentlightning.semconv.AGL_MESSAGE] span."""

    annotation_type: AnnotationType = "message"
    """Type of the annotation."""

    message: str
    """Message text."""


class ObjectAnnotation(Annotation):
    """An artifact that is parsed from a [object][agentlightning.semconv.AGL_OBJECT] span."""

    annotation_type: AnnotationType = "object"
    """Type of the annotation."""

    object: Any
    """The object payload."""


class ExceptionAnnotation(Annotation):
    """An exception that is parsed from an [exception][agentlightning.semconv.AGL_EXCEPTION] span."""

    annotation_type: AnnotationType = "exception"
    """Type of the annotation."""

    type: str
    """Type of the exception."""

    message: str
    """Message of the exception."""

    stacktrace: Optional[str] = None
    """Stacktrace of the exception."""


class OperationAnnotation(Annotation):
    """An operation that is parsed from an [operation][agentlightning.semconv.AGL_OPERATION] span."""

    annotation_type: AnnotationType = "operation"
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


class AnnotatedChatCompletionCall(ChatCompletionCall):
    """A chat completion call with annotations."""

    annotations: Sequence[Annotation]
    """Annotations for the chat completion call."""


# Algorithm-specific requirements

T_observation = TypeVar("T_observation")
T_action = TypeVar("T_action")


class Triplet(BaseModel, Generic[T_observation, T_action]):
    """A triplet of observation, action and reward."""

    observation: T_observation
    """Observation for the model input."""

    action: T_action
    """Action from the model output."""

    reward: Optional[float]
    """Reward of the model input."""

    done: bool
    """Whether it's the end of the trajectory."""

    raw_call: Union[AnnotatedChatCompletionCall, ChatCompletionCall]
    """Raw chat completion call."""


class Accumulation(BaseModel):
    """Accumulation from multiple triplets."""

    final_reward: Optional[float]
    """Single reward value for the entire sequence.

    An accumulation can only have one single final reward value.
    """

    raw_calls: Sequence[Union[AnnotatedChatCompletionCall, ChatCompletionCall]]
    """Raw chat completion calls. The order of the calls must be the same as the order of the token IDs."""


class TokenInput(BaseModel):
    """Token-based model input."""

    token_ids: Sequence[int]
    """Token IDs of the model input."""

    image_urls: Sequence[str]
    """A list of image URLs. Could be pointers to local files or base64-encoded images."""


class TokenOutput(BaseModel):
    """Token-based model output."""

    token_ids: Sequence[int]
    """Token IDs of the model output."""

    logprobs: Optional[Sequence[float]]
    """Log probabilities of the model output."""

    @model_validator(mode="after")
    def validate_logprobs(self) -> Self:
        if self.logprobs is not None:
            if len(self.logprobs) != len(self.token_ids):
                raise ValueError("Log probabilities must be the same length as the token IDs.")
        return self


class TokensTriplet(Triplet[TokenInput, TokenOutput]):
    """A triplet of token IDs for the input and output, useful for reinforcement learning.

    This is not a stable interface and the fields here highly depend on RL implementations.
    """


class TokensAccumulationDiagnosis(BaseModel):

    special_tokens_mismatch: bool
    """Whether there is a mismatch in special tokens."""

    non_special_tokens_mismatch: bool
    """Whether there is a mismatch in non-special tokens."""

    detokenized_text_mismatch: bool
    """Whether there is a mismatch in detokenized text."""

    image_urls_mismatch: bool
    """Whether there is a mismatch in image URLs."""

    accumulation_prev: TokensAccumulation
    """The previous accumulation which this triplet fails to match."""

    special_tokens_prev: Sequence[int]
    """Special tokens in the previous accumulation."""

    special_tokens_next: Sequence[int]
    """Special tokens in the next sequence."""

    detokenized_text_prev: str
    """Detokenized text in the previous accumulation."""

    detokenized_text_next: str
    """Detokenized text in the next sequence."""


class TokensAccumulation(Accumulation):
    """A sequence of token IDs that are accumulated from multiple model calls.

    Output is implied in the token IDs.
    """

    token_ids: Sequence[int]
    """Token IDs of the model input and output."""

    image_urls: Sequence[str]
    """Image URLs of the model input and output."""

    logprobs: Optional[Sequence[float]]
    """Log probabilities of the model output."""

    response_mask: Sequence[int]
    """Mask for the response tokens. Must a sequence of 0s and 1s, with 1s for the completion tokens and 0s for the prompt tokens."""

    diagnosis_info: Optional[TokensAccumulationDiagnosis]
    """Diagnosis information for token accumulation mismatches."""


class PromptCompletionTriplet(Triplet[CompletionCreateParams, ChatCompletion]):
    """A triplet of prompt and completion."""


class PromptCompletionAccumulation(Accumulation):
    """A conversation that is accumulated from multiple model calls."""

    messages: Sequence[ChatCompletionMessageParam]
    """Messages of the conversation."""

    tools: Optional[Sequence[ChatCompletionToolUnionParam]]
    """Tools provided for the conversation."""
