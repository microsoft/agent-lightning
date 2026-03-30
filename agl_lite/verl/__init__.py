# Copyright (c) Microsoft. All rights reserved.

"""VERL integration for agl-lite — adapted from Agent Lightning.

Only the daemon is exported by default (it guards heavy imports).
The trainer, dataset, and entrypoint modules require torch/verl/ray
and should be imported directly when those deps are available.
"""

from .daemon import *  # noqa: F401,F403
