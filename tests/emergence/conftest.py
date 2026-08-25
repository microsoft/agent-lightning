# Copyright (c) Microsoft. All rights reserved.

"""Shared fixtures for emergence module tests."""

from __future__ import annotations

# Patch Unix-only modules for Windows compatibility before any agentlightning imports
import platform
import sys
import types as _types
from unittest.mock import MagicMock

if platform.system() == "Windows":
    for _mod_name in ("fcntl", "pwd", "grp", "resource"):
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = _types.ModuleType(_mod_name)
    # gunicorn is deeply Unix-specific; mock the entire tree
    for _mod_name in [
        "gunicorn", "gunicorn.app", "gunicorn.app.base", "gunicorn.arbiter",
        "gunicorn.sock", "gunicorn.systemd", "gunicorn.util", "gunicorn.config",
        "gunicorn.errors", "gunicorn.http", "gunicorn.http.wsgi",
        "gunicorn.workers", "gunicorn.workers.base",
    ]:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = MagicMock()

import itertools
from typing import Any, Dict, List, Optional

import pytest

from agentlightning.adapter.triplet import TraceTree
from agentlightning.types import Span, Triplet

_SEQ = itertools.count()


def make_span(
    name: str = "test_span",
    *,
    rollout_id: str = "rollout-1",
    attempt_id: str = "attempt-1",
    parent_id: Optional[str] = None,
    span_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    start_time: float = 0.0,
    end_time: float = 1.0,
    attributes: Optional[Dict[str, Any]] = None,
) -> Span:
    """Create a minimal Span for testing."""
    return Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=next(_SEQ),
        name=name,
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        start_time=start_time,
        end_time=end_time,
        attributes=attributes or {},
    )


def make_tree(
    name: str = "root",
    children: Optional[List[TraceTree]] = None,
    *,
    start_time: float = 0.0,
    end_time: float = 1.0,
) -> TraceTree:
    """Create a TraceTree node for testing."""
    span = make_span(name=name, start_time=start_time, end_time=end_time)
    return TraceTree(id=span.span_id, span=span, children=children or [])


def make_diverse_trees(n: int = 10) -> List[TraceTree]:
    """Create n structurally diverse trace trees."""
    trees: List[TraceTree] = []
    tool_names = ["search", "analyze", "respond", "compute", "lookup", "transform"]
    for i in range(n):
        name = tool_names[i % len(tool_names)]
        child_count = (i % 3) + 1
        children = [
            make_tree(f"{name}_child_{j}", start_time=float(j), end_time=float(j + 1))
            for j in range(child_count)
        ]
        trees.append(make_tree(f"root_{name}", children=children))
    return trees


def make_uniform_trees(n: int = 10) -> List[TraceTree]:
    """Create n structurally identical trace trees."""
    trees: List[TraceTree] = []
    for _ in range(n):
        child = make_tree("openai.chat.completion", start_time=0.1, end_time=0.5)
        trees.append(make_tree("root", children=[child]))
    return trees


def make_triplet(reward: Optional[float] = None, **metadata: Any) -> Triplet:
    """Create a Triplet for testing."""
    return Triplet(
        prompt={"token_ids": [1, 2, 3]},
        response={"token_ids": [4, 5, 6]},
        reward=reward,
        metadata=metadata,
    )
