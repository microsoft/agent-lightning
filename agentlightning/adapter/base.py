# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Callable, Generic, Sequence, TypeVar, overload

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import Span
from agentlightning.types.adapter import BaseAdaptingSequence, BaseAdaptingSequenceItem

T_inv = TypeVar("T_inv")
T_from = TypeVar("T_from", contravariant=True)
T_to = TypeVar("T_to", covariant=True)
T_seq_from = TypeVar("T_seq_from", contravariant=True, bound=BaseAdaptingSequenceItem)
T_seq_to = TypeVar("T_seq_to", covariant=True, bound=BaseAdaptingSequenceItem)


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


class SequenceAdapter(
    Adapter[BaseAdaptingSequence[T_seq_from], BaseAdaptingSequence[T_seq_to]],
    Generic[T_seq_from, T_seq_to],
):
    """Base class for adapters that convert adapting sequences of data from one format to another.

    This class specializes [`Adapter`][agentlightning.Adapter] for working with
    [`AdaptingSequence`][agentlightning.AdaptingSequence] instances.
    """

    def adapt(self, source: BaseAdaptingSequence[T_seq_from]) -> BaseAdaptingSequence[T_seq_to]:
        return source.map(self.adapt_one)

    def adapt_one(self, source: T_seq_from) -> T_seq_to:
        raise NotImplementedError(f"{self.__class__.__name__}.adapt_one() is not implemented")


class Filter(Adapter[Sequence[T_inv], Sequence[T_inv]], Generic[T_inv]):
    """Filter items of type T to items of type T based on a predicate."""

    def __init__(self, predicate: Callable[[T_inv], bool]) -> None:
        self.predicate = predicate

    def adapt(self, source: Sequence[T_inv]) -> Sequence[T_inv]:
        return [item for item in source if self.predicate(item)]


class Sort(Adapter[Sequence[T_inv], Sequence[T_inv]], Generic[T_inv]):
    """Sort items of type T based on a key function."""

    def __init__(self, key: Callable[[T_inv], Any]) -> None:
        self.key = key

    def adapt(self, source: Sequence[T_inv]) -> Sequence[T_inv]:
        return sorted(source, key=self.key)


T_chain1 = TypeVar("T_chain1")
T_chain2 = TypeVar("T_chain2")
T_chain3 = TypeVar("T_chain3")
T_chain4 = TypeVar("T_chain4")
T_chain5 = TypeVar("T_chain5")
T_chain6 = TypeVar("T_chain6")
T_chain7 = TypeVar("T_chain7")
T_chain8 = TypeVar("T_chain8")
T_chain9 = TypeVar("T_chain9")


class Chain(Adapter[T_from, T_to]):
    """Chain multiple adapters together to form a single adapter.

    The output of each adapter is passed as input to the next adapter in the chain.
    """

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_chain4],
        adapter5: Adapter[T_chain4, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_chain4],
        adapter5: Adapter[T_chain4, T_chain5],
        adapter6: Adapter[T_chain5, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_chain4],
        adapter5: Adapter[T_chain4, T_chain5],
        adapter6: Adapter[T_chain5, T_chain6],
        adapter7: Adapter[T_chain6, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_chain4],
        adapter5: Adapter[T_chain4, T_chain5],
        adapter6: Adapter[T_chain5, T_chain6],
        adapter7: Adapter[T_chain6, T_chain7],
        adapter8: Adapter[T_chain7, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_chain4],
        adapter5: Adapter[T_chain4, T_chain5],
        adapter6: Adapter[T_chain5, T_chain6],
        adapter7: Adapter[T_chain6, T_chain7],
        adapter8: Adapter[T_chain7, T_chain8],
        adapter9: Adapter[T_chain8, T_to],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter1: Adapter[T_from, T_chain1],
        adapter2: Adapter[T_chain1, T_chain2],
        adapter3: Adapter[T_chain2, T_chain3],
        adapter4: Adapter[T_chain3, T_chain4],
        adapter5: Adapter[T_chain4, T_chain5],
        adapter6: Adapter[T_chain5, T_chain6],
        adapter7: Adapter[T_chain6, T_chain7],
        adapter8: Adapter[T_chain7, T_chain8],
        adapter9: Adapter[T_chain8, T_chain9],
        adapter10: Adapter[T_chain9, T_to],
        /,
    ) -> None: ...

    def __init__(self, adapter1: Adapter[Any, Any], *adapters: Adapter[Any, Any]) -> None:
        # Enforce that a Chain always has at least one adapter.
        self.adapters: tuple[Adapter[Any, Any], ...] = (adapter1, *adapters)

    def adapt(self, source: T_from) -> T_to:
        result: Any = source
        for idx, adapter in enumerate(self.adapters):
            try:
                result = adapter.adapt(result)
            except Exception as exc:
                raise RuntimeError(
                    f"Adapter chain failed at adapter index {idx} ({adapter.__class__.__name__}). See inner exception for details."
                ) from exc
        return result


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
