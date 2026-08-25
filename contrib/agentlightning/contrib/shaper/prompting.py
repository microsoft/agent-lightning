# Copyright (c) Microsoft. All rights reserved.

"""Prompt loading and strict JSON-response parsing for SHAPER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a version-controlled SHAPER role prompt."""

    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse one JSON object, tolerating a single surrounding Markdown fence."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline >= 0:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    value: Any = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object from SHAPER role model.")
    return cast(Dict[str, Any], value)
