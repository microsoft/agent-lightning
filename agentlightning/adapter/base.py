# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Callable, Generic, Sequence, TypeVar

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import Span

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")


class Adapter(Generic[T_from, T_to]):
    """Base class for synchronous adapters that convert data from one format to another.

    The class defines a minimal protocol so that adapters can be treated like callables while
    still allowing subclasses to supply the concrete transformation logic.

    !!! note
        Subclasses must override [`adapt()`][agentlightning.Adapter.adapt] to provide
        the actual conversion.

    Type Variables:

        T_from: Source data type supplied to the adapter.

        T_to: Target data type produced by the adapter.

    Examples:
        >>> class IntToStrAdapter(Adapter[int, str]):
        ...     def adapt(self, source: int) -> str:
        ...         return str(source)
        ...
        >>> adapter = IntToStrAdapter()
        >>> adapter(42)
        '42'
    """

    def __call__(self, source: T_from, /) -> T_to:
        """Convert the data to the target format.

        This method delegates to [`adapt()`][agentlightning.Adapter.adapt] so that an
        instance of [`Adapter`][agentlightning.Adapter] can be used like a standard
        function.

        Args:
            source: Input data in the source format.

        Returns:
            Data converted to the target format.
        """
        return self.adapt(source)

    def adapt(self, source: T_from, /) -> T_to:
        """Convert the data to the target format.

        Subclasses must override this method with the concrete transformation logic. The base
        implementation raises `NotImplementedError` to make the requirement explicit.

        Args:
            source: Input data in the source format.

        Returns:
            Data converted to the target format.
        """
        raise NotImplementedError("Adapter.adapt() is not implemented")


class SequenceAdapter(Adapter[Sequence[T_from], Sequence[T_to]], Generic[T_from, T_to]):
    """Base class for adapters that convert sequences of data from one format to another.

    This class specializes [`Adapter`][agentlightning.Adapter] for working with sequences of data.
    """

    def adapt(self, source: Sequence[T_from]) -> Sequence[T_to]:
        return [self.adapt_one(item) for item in source]

    def adapt_one(self, source: T_from) -> T_to:
        raise NotImplementedError("SequenceAdapter.adapt_one() is not implemented")


class Filter(Adapter[Sequence[T_from], Sequence[T_from]], Generic[T_from]):
    """Filter items of type T to items of type T based on a predicate."""

    def __init__(self, predicate: Callable[[T_from], bool]) -> None:
        self.predicate = predicate

    def adapt(self, source: Sequence[T_from]) -> Sequence[T_from]:
        return [item for item in source if self.predicate(item)]


class Sort(Adapter[Sequence[T_from], Sequence[T_from]], Generic[T_from]):
    """Sort items of type T based on a key function."""

    def __init__(self, key: Callable[[T_from], Any]) -> None:
        self.key = key

    def adapt(self, source: Sequence[T_from]) -> Sequence[T_from]:
        return sorted(source, key=self.key)


class OtelTraceAdapter(Adapter[Sequence[ReadableSpan], T_to], Generic[T_to]):
    """Base class for adapters that convert OpenTelemetry trace spans into other formats.

    This specialization of [`Adapter`][agentlightning.Adapter] expects a list of
    `opentelemetry.sdk.trace.ReadableSpan` instances and produces any target format, such as
    reinforcement learning trajectories, structured logs, or analytics-ready payloads.

    Examples:
        >>> class TraceToDictAdapter(OtelTraceAdapter[dict]):
        ...     def adapt(self, spans: List[ReadableSpan]) -> dict:
        ...         return {"count": len(spans)}
        ...
        >>> adapter = TraceToDictAdapter()
        >>> adapter([span1, span2])
        {'count': 2}
    """


class TraceAdapter(Adapter[Sequence[Span], T_to], Generic[T_to]):
    """Base class for adapters that convert trace spans into other formats.

    This class specializes [`Adapter`][agentlightning.Adapter] for working with
    [`Span`][agentlightning.Span] instances emitted by Agent Lightning instrumentation.
    Subclasses receive entire trace slices and return a format suited for the downstream consumer,
    for example reinforcement learning training data or observability metrics.
    """
