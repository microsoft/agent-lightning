# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Sequence, Tuple


class _CounterChild:
    def __init__(self) -> None:
        self.value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


class _HistogramChild:
    def __init__(self) -> None:
        self.values: List[float] = []

    def observe(self, value: float) -> None:
        self.values.append(value)


class PrometheusStub(types.ModuleType):
    """Minimal prometheus_client replacement for unit tests."""

    def __init__(self, real_client: Any) -> None:
        super().__init__("prometheus_client")
        self.counter_instances: List[_PromCounter] = []
        self.histogram_instances: List[_PromHistogram] = []
        self.real_client = real_client

        class CollectorRegistry:
            pass

        class _Multiprocess:
            def __init__(self) -> None:
                self.registry: Optional[CollectorRegistry] = None

            def MultiProcessCollector(self, registry: CollectorRegistry) -> None:
                self.registry = registry

        self.CollectorRegistry = CollectorRegistry
        self.REGISTRY = CollectorRegistry()
        self.multiprocess = _Multiprocess()

        self.Counter = _PromCounterFactory(self)
        self.Histogram = _PromHistogramFactory(self)
        self.make_asgi_app = self._make_asgi_app

    def _make_asgi_app(self, *args: Any, **kwargs: Any) -> Any:
        return self.real_client.make_asgi_app(*args, **kwargs)


class _PromCounterFactory:
    def __init__(self, owner: PrometheusStub) -> None:
        self._owner = owner

    def __call__(self, name: str, doc: str, labelnames: Sequence[str]) -> _PromCounter:
        counter = _PromCounter(name, doc, labelnames)
        counter._register(self._owner.counter_instances)  # pyright: ignore[reportPrivateUsage]
        return counter


class _PromHistogramFactory:
    def __init__(self, owner: PrometheusStub) -> None:
        self._owner = owner

    def __call__(
        self,
        name: str,
        doc: str,
        labelnames: Sequence[str],
        buckets: Sequence[float] | None = None,
    ) -> _PromHistogram:
        histogram = _PromHistogram(name, doc, labelnames, buckets or ())
        histogram._register(self._owner.histogram_instances)  # pyright: ignore[reportPrivateUsage]
        return histogram


class _PromCounter:
    def __init__(self, name: str, doc: str, labelnames: Sequence[str]) -> None:
        self.name = name
        self.doc = doc
        self.labelnames = tuple(labelnames)
        self.default = _CounterChild()
        self.children: Dict[Tuple[Tuple[str, str], ...], _CounterChild] = {}

    def _register(self, sink: List["_PromCounter"]) -> None:
        sink.append(self)

    def labels(self, **kwargs: str) -> _CounterChild:
        key = tuple(sorted(kwargs.items()))
        return self.children.setdefault(key, _CounterChild())

    def inc(self, amount: float = 1.0) -> None:
        self.default.inc(amount)


class _PromHistogram:
    def __init__(self, name: str, doc: str, labelnames: Sequence[str], buckets: Sequence[float]) -> None:
        self.name = name
        self.doc = doc
        self.labelnames = tuple(labelnames)
        self.buckets = tuple(buckets)
        self.default = _HistogramChild()
        self.children: Dict[Tuple[Tuple[str, str], ...], _HistogramChild] = {}

    def _register(self, sink: List["_PromHistogram"]) -> None:
        sink.append(self)

    def labels(self, **kwargs: str) -> _HistogramChild:
        key = tuple(sorted(kwargs.items()))
        return self.children.setdefault(key, _HistogramChild())

    def observe(self, value: float) -> None:
        self.default.observe(value)


def make_prometheus_stub() -> PrometheusStub:
    """Factory helper for tests."""
    import prometheus_client

    return PrometheusStub(prometheus_client)
