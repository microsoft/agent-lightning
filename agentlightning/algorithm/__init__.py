# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Algorithm
from .decorator import algo
from .fast import Baseline, FastAlgorithm

if TYPE_CHECKING:
    from .apo import APO as APOType
    from .gepa import GEPA as GEPAType
    from .verl import VERL as VERLType

__all__ = ["Algorithm", "algo", "FastAlgorithm", "Baseline", "APO", "VERL", "GEPA"]

# Shortcuts for usages like algo.APO(...)


def APO(*args: Any, **kwargs: Any) -> APOType[Any]:
    from .apo import APO as APOImplementation

    return APOImplementation(*args, **kwargs)


def VERL(*args: Any, **kwargs: Any) -> VERLType:
    from .verl import VERL as VERLImplementation

    return VERLImplementation(*args, **kwargs)


def GEPA(*args: Any, **kwargs: Any) -> GEPAType:
    from .gepa import GEPA as GEPAImplementation

    return GEPAImplementation(*args, **kwargs)
