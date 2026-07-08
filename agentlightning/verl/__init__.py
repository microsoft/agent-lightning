# Copyright (c) Microsoft. All rights reserved.

"""This package contains a *hacky* integration of VERL with Agent Lightning."""

from .trajectory import *

try:
    from .daemon import *
    from .dataset import *
    from .entrypoint import *
    from .trainer import *
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "torch", "tensordict", "verl", "ray", "vllm"}:
        raise
