# Copyright (c) Microsoft. All rights reserved.

import warnings

from .emitter.reward import *  # noqa: F401,F403

warnings.warn("agentlightning.reward is deprecated. Please use agentlightning.emitter instead.")
