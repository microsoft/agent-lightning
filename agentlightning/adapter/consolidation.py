# Copyright (c) Microsoft. All rights reserved.

"""Filter/aggregate multiple objects into a single object.

Opinionated towards which objects to keep and how to aggregate them.
"""

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from agentlightning.types.adapter import (
    AccumulatedTokenSequence,
    AnnotatedChatCompletionCall,
    Annotation,
    ChatCompletionCall,
    TokenInputOutputTriplet,
    Tree,
)
