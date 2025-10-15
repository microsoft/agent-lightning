# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseAlgorithm
from .decorator import algo
from .fast import Baseline, FastAlgorithm

if TYPE_CHECKING:
    from .apo import APO as APOType
    from .verl import VERL as VERLType

__all__ = ["BaseAlgorithm", "algo", "FastAlgorithm", "Baseline", "APO", "VERL"]

# Shortcuts for usages like algo.APO(...)


def APO(*args: Any, **kwargs: Any) -> APOType[Any]:
    from .apo import APO as APOImplementation

    return APOImplementation(*args, **kwargs)


def VERL(*args: Any, **kwargs: Any) -> VERLType:
    from .verl import VERL as VERLImplementation

    return VERLImplementation(*args, **kwargs)
