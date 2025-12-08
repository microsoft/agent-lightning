# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any


class SampleBase:
    pass


class SampleComponent(SampleBase):
    def __init__(self, required: int, optional: int = 7) -> None:
        self.required = required
        self.optional = optional


class SimpleComponent(SampleBase):
    def __init__(self) -> None:
        self.created = True


class ComponentWithoutOptional(SampleBase):
    def __init__(self, value: int) -> None:
        self.value = value


class ComponentWithKwargs(SampleBase):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class Unrelated:
    pass
