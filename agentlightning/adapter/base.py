# Copyright (c) Microsoft. All rights reserved.

from typing import Generic, List, TypeVar

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import Span

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")


class Adapter(Generic[T_from, T_to]):
    """Base class for synchronous adapters that convert data from one format to another.

    This class defines a simple protocol for transformation:

    - The `__call__` method makes adapters callable, so they can be used like functions.
    - Subclasses must implement the `adapt` method to define the actual conversion logic.

    Type parameters:

    - T_from: The source data type (input).
    - T_to: The target data type (output).

    Example:

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

        This method delegates to `adapt` and allows the adapter
        to be invoked as a function.

        Args:
            source: Input data in the source format.

        Returns:
            Data converted to the target format.
        """
        return self.adapt(source)

    def adapt(self, source: T_from, /) -> T_to:
        """Convert the data to the target format.

        Subclasses should override this method with the concrete
        transformation logic.

        Args:
            source: Input data in the source format.

        Returns:
            Data converted to the target format.
        """
        raise NotImplementedError("Adapter.adapt() is not implemented")


class OtelTraceAdapter(Adapter[List[ReadableSpan], T_to], Generic[T_to]):
    """Base class for adapters that convert OpenTelemetry trace spans into other formats.

    This class specializes `Adapter` for working with OpenTelemetry `ReadableSpan`
    objects. It expects a list of spans as input and produces a custom target format
    (e.g., reinforcement learning training data, SFT datasets, logs, metrics).

    Subclasses should override `adapt` to define the desired conversion.

    Type parameters:
        T_to: The target data type that spans should be converted into.

    Example:
        >>> class TraceToDictAdapter(TraceAdapter[dict]):
        ...     def adapt(self, spans: List[ReadableSpan]) -> dict:
        ...         return {"count": len(spans)}
        ...
        >>> adapter = TraceToDictAdapter()
        >>> adapter([span1, span2])
        {'count': 2}
    """


class TraceAdapter(Adapter[List[Span], T_to], Generic[T_to]):
    """Base class for adapters that convert trace spans into other formats.

    This class specializes `Adapter` for working with trace spans. It expects a list of
    Agent-lightning spans as input and produces a custom target format
    (e.g., reinforcement learning training data, SFT datasets, logs, metrics).
    """
