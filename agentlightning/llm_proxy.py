# Copyright (c) Microsoft. All rights reserved.

from .types import NamedResources


class LlmProxy:

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def as_resources(self) -> NamedResources:
        ...
