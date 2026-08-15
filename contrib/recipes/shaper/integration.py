# Copyright (c) Microsoft. All rights reserved.

"""Small helpers for connecting an embodied agent to the SHAPER trace contract."""

from __future__ import annotations

from typing import Any, Callable, Dict

from agentlightning.contrib.shaper import PythonHarnessValidator
from agentlightning.types import NamedResources, PromptTemplate


def get_artifact_text(
    resources: NamedResources,
    *,
    skill_resource_name: str = "skill",
    harness_resource_name: str = "harness",
) -> tuple[str, str]:
    """Extract skill text and harness source from an AGL resource bundle."""

    skill = resources.get(skill_resource_name)
    harness = resources.get(harness_resource_name)
    if not isinstance(skill, PromptTemplate):
        raise TypeError(f"{skill_resource_name!r} must be a PromptTemplate resource.")
    if not isinstance(harness, PromptTemplate):
        raise TypeError(f"{harness_resource_name!r} must be a PromptTemplate resource.")
    return skill.template, harness.template


def load_context_builder(
    source: str,
    *,
    validator: PythonHarnessValidator | None = None,
) -> Callable[[list[Dict[str, Any]]], Any]:
    """Validate a context builder and return its restricted-process callable."""

    effective_validator = validator or PythonHarnessValidator()
    runtime = effective_validator.runtime(source)
    return runtime
