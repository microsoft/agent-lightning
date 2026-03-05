# Copyright (c) Microsoft. All rights reserved.

"""Codec for converting between Agent Lightning resources and GEPA candidates."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from agentlightning.types import NamedResources, PromptTemplate

logger = logging.getLogger(__name__)


class PromptResourceCodec:
    """Bidirectional converter between AGL `NamedResources` and GEPA candidate dicts.

    GEPA candidates are plain ``dict[str, str]`` mappings from component names
    to text content.  Agent Lightning resources are ``NamedResources`` mappings
    from resource names to typed ``Resource`` objects.  This codec bridges the
    two representations for single-`PromptTemplate` optimization, preserving
    the template engine across round-trips.

    Args:
        resource_name: The key in `NamedResources` that holds the optimizable
            `PromptTemplate`.
        engine: The template engine to use when reconstructing `PromptTemplate`
            objects from GEPA candidate text (e.g. ``"f-string"``, ``"jinja"``).
    """

    def __init__(self, resource_name: str, engine: str) -> None:
        self.resource_name = resource_name
        self.engine = engine

    def resources_to_candidate(self, resources: NamedResources) -> Dict[str, str]:
        """Extract optimizable text from Agent Lightning resources.

        Args:
            resources: Named resource mapping containing the target `PromptTemplate`.

        Returns:
            A GEPA candidate dict mapping the resource name to the template text.

        Raises:
            KeyError: If the configured resource name is not present.
            TypeError: If the resource is not a `PromptTemplate`.
        """
        resource = resources[self.resource_name]
        if not isinstance(resource, PromptTemplate):
            raise TypeError(f"Resource '{self.resource_name}' is not a PromptTemplate, got {type(resource).__name__}")
        return {self.resource_name: resource.template}

    def candidate_to_resources(self, candidate: Dict[str, str]) -> NamedResources:
        """Rebuild Agent Lightning resources from a GEPA candidate dict.

        Args:
            candidate: GEPA candidate mapping component names to text.

        Returns:
            A `NamedResources` mapping containing the reconstructed `PromptTemplate`.

        Raises:
            KeyError: If the configured resource name is not in the candidate.
        """
        template_text = candidate[self.resource_name]
        return {self.resource_name: PromptTemplate(template=template_text, engine=self.engine)}  # type: ignore[arg-type]

    @classmethod
    def from_initial_resources(
        cls,
        resources: NamedResources,
        resource_name: str | None = None,
    ) -> Tuple[PromptResourceCodec, Dict[str, str]]:
        """Auto-detect the first `PromptTemplate` and build a codec plus seed candidate.

        When ``resource_name`` is provided, that specific resource is used.
        Otherwise, the first `PromptTemplate` found in ``resources`` is selected.

        Args:
            resources: Initial named resources from the trainer.
            resource_name: Explicit resource key to use. When ``None``,
                auto-detects the first `PromptTemplate`.

        Returns:
            A tuple of ``(codec, seed_candidate)`` ready for GEPA's ``optimize()``.

        Raises:
            ValueError: If no `PromptTemplate` is found in resources, or the
                specified ``resource_name`` does not exist or is not a
                `PromptTemplate`.
        """
        if resource_name is not None:
            if resource_name not in resources:
                raise ValueError(f"Resource '{resource_name}' not found in initial_resources")
            resource = resources[resource_name]
            if not isinstance(resource, PromptTemplate):
                raise ValueError(f"Resource '{resource_name}' is not a PromptTemplate, got {type(resource).__name__}")
            detected_name = resource_name
            detected_resource = resource
        else:
            detected_name = None
            detected_resource = None
            for name, resource in resources.items():
                if isinstance(resource, PromptTemplate):
                    detected_name = name
                    detected_resource = resource
                    break
            if detected_name is None or detected_resource is None:
                raise ValueError("No PromptTemplate found in initial_resources")

        logger.info(
            "Using resource '%s' (engine=%s) for GEPA optimization",
            detected_name,
            detected_resource.engine,
        )
        codec = cls(resource_name=detected_name, engine=detected_resource.engine)
        seed_candidate = {detected_name: detected_resource.template}
        return codec, seed_candidate
