# Copyright (c) Microsoft. All rights reserved.

"""Fresh-process bridge to ESI-Bench's official active-exploration pipeline."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from openai import OpenAI

from ..common import (
    first_json_object,
    image_part,
    normalize_content,
    path_data_url,
    sanitized_action_result,
    strip_thinking,
    text_part,
)
from ..harness_bridge import HarnessBridgeClient, HarnessBridgeError
from .contracts import (
    check_behavior_source,
    check_omnigibson_install,
    check_upstream_source,
    validate_skill,
)
from .dataset import load_question_row, resolve_canonical_question

logger = logging.getLogger(__name__)

GENERIC_JSON_INSTRUCTION = "Return exactly one valid JSON object and nothing else."

_deferred_simulator_shutdown: Callable[[], Any] | None = None


class PlannerRequestError(RuntimeError):
    """The frozen planner endpoint failed before an action could be produced."""


class OfficialRunnerError(RuntimeError):
    """The official simulator or task runner failed independently of artifacts."""


def _render_skill(skill: str, task_prompt: str) -> str:
    """Wrap one official task prompt with the validated reusable skill."""

    if skill.count("{task_prompt}") != 1:
        raise ValueError("ESI-Bench skill must contain {task_prompt} exactly once.")
    return skill.replace("{task_prompt}", task_prompt, 1)


def _jpeg_data_url(image: Any, *, max_side: int = 768, quality: int = 90) -> str:
    """Encode a bounded deterministic BGR array for harness transport."""

    cv2 = cast(Any, importlib.import_module("cv2"))
    numpy = cast(Any, importlib.import_module("numpy"))
    encoded_image = numpy.asarray(image)
    height, width = encoded_image.shape[:2]
    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        encoded_image = cv2.resize(encoded_image, target, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg",
        numpy.ascontiguousarray(encoded_image),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise ValueError("OpenCV could not encode the official ESI-Bench RGB.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _pixel_quality(image: Any) -> dict[str, float]:
    """Score one crop using rendered pixels only.

    The three terms mirror the evidence-quality signals described by SHAPER:
    grayscale contrast, finite-difference edge density, and Laplacian
    sharpness. They carry no semantic label or simulator metadata.
    """

    numpy = cast(Any, importlib.import_module("numpy"))
    pixels = numpy.asarray(image)
    if pixels.size == 0:
        return {"contrast": 0.0, "edge_density": 0.0, "laplacian_sharpness": 0.0, "score": 0.0}
    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        gray = 0.114 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.299 * pixels[..., 2]
    elif pixels.ndim == 3:
        gray = pixels.mean(axis=2)
    else:
        gray = pixels
    gray = numpy.asarray(gray, dtype=numpy.float32)
    row_stride = max(1, (int(gray.shape[0]) + 71) // 72)
    column_stride = max(1, (int(gray.shape[1]) + 127) // 128)
    gray = gray[::row_stride, ::column_stride][:72, :128]
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return {"contrast": 0.0, "edge_density": 0.0, "laplacian_sharpness": 0.0, "score": 0.0}
    contrast = min(1.5, float(gray.std()) / 55.0)
    horizontal = numpy.abs(numpy.diff(gray, axis=1))
    vertical = numpy.abs(numpy.diff(gray, axis=0))
    edge_density = 0.5 * (float((horizontal > 16.0).mean()) + float((vertical > 16.0).mean()))
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    sharpness = min(1.5, float(laplacian.var()) / 900.0)
    score = contrast + 5.0 * edge_density + 0.35 * sharpness
    return {
        "contrast": round(contrast, 6),
        "edge_density": round(edge_density, 6),
        "laplacian_sharpness": round(sharpness, 6),
        "score": round(score, 6),
    }


def _focus_crop_candidates(image: Any) -> list[dict[str, Any]]:
    """Select one horizontal band and one overlapping tile deterministically."""

    height, width = image.shape[:2]
    candidates: list[tuple[str, str, Any, dict[str, float]]] = []
    band_height = min(height, max(1, round(height * 0.28)))
    for index, fraction in enumerate((0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72), start=1):
        top = min(height - band_height, round(height * fraction))
        crop = image[top : top + band_height, 0:width]
        candidates.append(("horizontal_band", f"horizontal_band_{index}", crop, _pixel_quality(crop)))

    tile_width = min(width, max(1, round(width * 0.62)))
    tile_height = min(height, max(1, round(height * 0.55)))
    tile_index = 0
    for y_fraction in (0.0, 0.225, 0.45):
        for x_fraction in (0.0, 0.19, 0.38):
            tile_index += 1
            left = min(width - tile_width, round(width * x_fraction))
            top = min(height - tile_height, round(height * y_fraction))
            crop = image[top : top + tile_height, left : left + tile_width]
            candidates.append(("overlapping_tile", f"overlapping_tile_{tile_index}", crop, _pixel_quality(crop)))

    selected: list[dict[str, Any]] = []
    for family in ("horizontal_band", "overlapping_tile"):
        family_candidates = [candidate for candidate in candidates if candidate[0] == family]
        _, region, crop, quality = max(
            family_candidates,
            key=lambda candidate: (float(candidate[3]["score"]), candidate[1]),
        )
        selected.append(
            {
                "source": "deterministic_pixel_crop",
                "region": region,
                "quality": quality,
                "image": image_part(_jpeg_data_url(crop, max_side=512)),
            }
        )
    return selected


def _visual_signature(image: Any) -> list[int]:
    """Return a tiny pixel-only signature for bounded keyframe diversity."""

    numpy = cast(Any, importlib.import_module("numpy"))
    pixels = numpy.asarray(image)
    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        gray = 0.114 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.299 * pixels[..., 2]
    elif pixels.ndim == 3:
        gray = pixels.mean(axis=2)
    else:
        gray = pixels
    gray = numpy.asarray(gray, dtype=numpy.float32)
    if gray.size == 0:
        return [0] * 64
    rows = numpy.rint(numpy.linspace(0, gray.shape[0] - 1, 8)).astype(int)
    columns = numpy.rint(numpy.linspace(0, gray.shape[1] - 1, 8)).astype(int)
    reduced = numpy.clip(numpy.rint(gray[numpy.ix_(rows, columns)]), 0, 255).astype(int)
    return [int(value) for value in reduced.reshape(-1).tolist()]


def _grid_overlay(image: Any) -> dict[str, Any]:
    """Overlay a deterministic GRID1000 coordinate system on official pixels."""

    cv2 = cast(Any, importlib.import_module("cv2"))
    numpy = cast(Any, importlib.import_module("numpy"))
    grid = numpy.asarray(image).copy()
    height, width = grid.shape[:2]
    line_color = (214, 160, 64)
    text_color = (255, 255, 255)
    shadow_color = (24, 24, 24)
    thickness = max(1, round(max(width, height) / 900))
    font_scale = max(0.35, min(width, height) / 1300.0)
    for value in range(0, 1001, 100):
        x = min(width - 1, round((width - 1) * value / 1000.0))
        y = min(height - 1, round((height - 1) * value / 1000.0))
        major = value % 250 == 0
        cv2.line(grid, (x, 0), (x, height - 1), line_color, thickness + int(major))
        cv2.line(grid, (0, y), (width - 1, y), line_color, thickness + int(major))
        if major:
            x_origin = min(max(2, x + 3), max(2, width - 46))
            y_origin = min(max(14, y + 14), max(14, height - 4))
            label = str(value)
            cv2.putText(
                grid,
                label,
                (x_origin + 1, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                shadow_color,
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                grid,
                label,
                (x_origin, 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                grid,
                label,
                (3, y_origin + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                shadow_color,
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                grid,
                label,
                (2, y_origin),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
    return {
        "source": "deterministic_pixel_overlay",
        "coordinate_system": "GRID1000: x=0..1000 left-to-right, y=0..1000 top-to-bottom",
        "image": image_part(_jpeg_data_url(grid)),
    }


def _needs_geometry_grid(task_prompt: str) -> bool:
    """Select pixel overlays from the official visible task contract only."""

    text = task_prompt.lower()
    return bool(
        any(term in text for term in ("triangle", "equilateral", "isosceles", "collinear"))
        or re.search(r"\b(?:in a|straight) line\b", text)
    )


class _ObservableImageCache:
    """Create full-frame and pixel-ranked crop payloads from official RGBs."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._grid_cache: dict[str, dict[str, Any]] = {}

    def observation(self, path: Path, *, include_grid: bool = False) -> dict[str, Any]:
        key = str(path.resolve())
        cached = self._cache.get(key)
        if cached is None:
            if not path.is_file():
                raise FileNotFoundError(f"Official ESI-Bench RGB does not exist: {path}")
            try:
                cv2 = cast(Any, importlib.import_module("cv2"))
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("OpenCV could not decode the image.")
                height, width = image.shape[:2]
                value: dict[str, Any] = {
                    "source": "official_rgb",
                    "pixel_size": {"width": int(width), "height": int(height)},
                    "pixel_quality": _pixel_quality(image),
                    "visual_signature": {
                        "kind": "8x8_grayscale",
                        "values": _visual_signature(image),
                    },
                    "full_frame": image_part(_jpeg_data_url(image)),
                    "focus_crops": _focus_crop_candidates(image),
                }
            except Exception as exc:
                logger.warning("Falling back to the original RGB bytes for %s: %s", path, exc)
                value = {
                    "source": "official_rgb",
                    "pixel_size": None,
                    "pixel_quality": None,
                    "visual_signature": None,
                    "full_frame": image_part(path_data_url(path)),
                    "focus_crops": [],
                }
            self._cache[key] = value
            cached = value
        output = cast(dict[str, Any], json.loads(json.dumps(cached)))
        if include_grid:
            grid = self._grid_cache.get(key)
            if grid is None:
                try:
                    cv2 = cast(Any, importlib.import_module("cv2"))
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if image is not None:
                        grid = _grid_overlay(image)
                        self._grid_cache[key] = grid
                except Exception as exc:
                    logger.warning("Could not construct GRID1000 overlay for %s: %s", path, exc)
            if grid is not None:
                output["grid_overlay"] = cast(
                    dict[str, Any],
                    json.loads(json.dumps(grid)),
                )
        return output


def _extra_paths(item: Mapping[str, Any]) -> list[Path]:
    output: list[Path] = []
    raw = item.get("extra_image_paths")
    if isinstance(raw, list):
        for value in cast(list[Any], raw):
            if isinstance(value, (str, Path)):
                output.append(Path(value))
    return output


def _without_inline_pixels(value: Any) -> Any:
    """Keep harness-input provenance while avoiding duplicate base64 payloads."""

    if isinstance(value, list):
        return [_without_inline_pixels(item) for item in cast(list[Any], value)]
    if not isinstance(value, dict):
        return value
    mapping = cast(dict[str, Any], value)
    if mapping.get("type") == "image_url" and isinstance(mapping.get("image_url"), dict):
        return {"type": "image_url", "image_url": {"url": "<observable pixels>"}}
    return {str(key): _without_inline_pixels(item) for key, item in mapping.items()}


def _decoded_json_field(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class _HarnessCollector:
    """Route every official planner context through one restricted harness."""

    def __init__(
        self,
        skill: str,
        build_context: Callable[[list[dict[str, Any]]], Any],
        *,
        max_steps: int,
    ) -> None:
        self.skill = skill
        self.build_context = build_context
        self.max_steps = max_steps
        self.images = _ObservableImageCache()
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.reference_images: list[dict[str, Any]] = []
        self._latest_past_records: list[dict[str, Any]] = []
        self._latest_history_index: int | None = None

    def _past_record(
        self,
        image_dir: Path,
        item: Mapping[str, Any],
        *,
        include_grid: bool,
    ) -> dict[str, Any]:
        image_path = image_dir / str(item.get("image", ""))
        extras = [
            self.images.observation(path, include_grid=include_grid) for path in _extra_paths(item) if path.is_file()
        ]
        action_result = sanitized_action_result(item.get("action_result"))
        return {
            "record_kind": "past",
            "step": int(item.get("step", 0)),
            "action": str(item.get("action", "")),
            "answer": str(item.get("answer", "")),
            "confidence": float(item.get("confidence", 0.0)),
            "reasoning": str(item.get("reasoning", "")),
            "action_result": action_result,
            "action_result_text": json.dumps(action_result, ensure_ascii=True),
            "observation": self.images.observation(image_path, include_grid=include_grid),
            "extra_observations": extras,
        }

    def __call__(
        self,
        image_path: Path,
        history: list[dict[str, Any]],
        prompt: str,
        reference_image_paths: list[Path] | None = None,
        reference_image_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        references = list(reference_image_paths or [])
        if reference_image_path is not None:
            references.append(reference_image_path)
        include_grid = _needs_geometry_grid(prompt)
        records = [self._past_record(image_path.parent, item, include_grid=include_grid) for item in history]
        reference_count = len(references)
        current: dict[str, Any] = {
            "record_kind": "current",
            "step": len(history) + 1,
            "max_steps": self.max_steps,
            "remaining_steps": max(0, self.max_steps - len(history)),
            "task_instruction": prompt,
            "observation": self.images.observation(image_path, include_grid=include_grid),
            "reference_observations": [
                {
                    "label": (
                        "[QUESTION REFERENCE IMAGE - dataset render]"
                        if reference_count == 1
                        else "[QUESTION REFERENCE IMAGE " + str(index) + "]"
                    ),
                    **self.images.observation(path, include_grid=include_grid),
                }
                for index, path in enumerate(references, start=1)
                if path.is_file()
            ],
        }
        records.append(current)
        if not self.reference_images:
            self.reference_images = [
                {
                    "label": str(reference.get("label", "QUESTION REFERENCE IMAGE")),
                    "image": cast(dict[str, Any], reference["full_frame"]),
                }
                for reference in cast(list[dict[str, Any]], current["reference_observations"])
                if isinstance(reference.get("full_frame"), dict)
            ]
        context = normalize_content(self.build_context(records))
        self._latest_past_records = records[:-1]
        self._latest_history_index = len(history)
        current_observation = cast(dict[str, Any], current["observation"])
        self.snapshots.setdefault(
            len(history),
            {
                "task_instruction": prompt,
                "harness_input": _without_inline_pixels(records),
                "context_payload": context,
                "observation_before": [current_observation["full_frame"]],
            },
        )
        # Match the official collector ordering: visual/history context first,
        # then the authoritative task prompt (wrapped by the selected skill).
        return [*context, text_part(_render_skill(self.skill, prompt))]

    def route_auxiliary(
        self,
        contents: Sequence[Any],
        task_prompt: str,
    ) -> list[dict[str, Any]]:
        """Route one audited task-specific call through the selected harness.

        The pinned inclined-plane hook supplies only ordered official RGB paths
        and visible text. They remain in that order, but RGBs receive the same
        pixel-only derivatives as primary observations so an evolved harness
        can select evidence instead of inheriting an unchangeable side path.
        """

        include_grid = _needs_geometry_grid(task_prompt)
        sequence: list[dict[str, Any]] = []
        for index, item in enumerate(contents):
            if isinstance(item, Path):
                sequence.append(
                    {
                        "content_kind": "observation",
                        "sequence_index": index,
                        "observation": self.images.observation(item, include_grid=include_grid),
                    }
                )
            elif isinstance(item, dict):
                mapping = cast(dict[str, Any], item)
                if mapping.get("type") == "text" and isinstance(mapping.get("text"), str):
                    sequence.append(
                        {
                            "content_kind": "text",
                            "sequence_index": index,
                            "text": str(mapping["text"]),
                        }
                    )
                elif mapping.get("type") == "image_url" and isinstance(mapping.get("image_url"), dict):
                    sequence.append(
                        {
                            "content_kind": "content_part",
                            "sequence_index": index,
                            "part": mapping,
                        }
                    )
                else:
                    sequence.append(
                        {
                            "content_kind": "text",
                            "sequence_index": index,
                            "text": str(mapping),
                        }
                    )
            else:
                sequence.append(
                    {
                        "content_kind": "text",
                        "sequence_index": index,
                        "text": str(item),
                    }
                )

        history_index = self._latest_history_index
        step = (history_index + 1) if history_index is not None else 1
        current: dict[str, Any] = {
            "record_kind": "current",
            "call_kind": "auxiliary_post_action",
            "step": step,
            "max_steps": self.max_steps,
            "remaining_steps": max(0, self.max_steps - (history_index or 0)),
            "task_instruction": task_prompt,
            "observable_sequence": sequence,
            "reference_observations": [],
        }
        records = [*self._latest_past_records, current]
        context = normalize_content(self.build_context(records))

        if history_index is not None:
            snapshot = self.snapshots.setdefault(
                history_index,
                {
                    "task_instruction": task_prompt,
                    "harness_input": _without_inline_pixels(records),
                    "context_payload": [],
                    "observation_before": [],
                },
            )
            existing = snapshot.get("context_payload")
            existing_contexts = cast(dict[str, Any], existing) if isinstance(existing, dict) else None
            call_contexts: dict[str, Any]
            if existing_contexts is not None and existing_contexts.get("call_contexts_version") == 1:
                call_contexts = existing_contexts
            else:
                call_contexts = {
                    "call_contexts_version": 1,
                    "primary": existing,
                    "auxiliary_post_action": [],
                }
                snapshot["context_payload"] = call_contexts
            auxiliary_value: Any = call_contexts.get("auxiliary_post_action")
            auxiliary: list[Any]
            if isinstance(auxiliary_value, list):
                auxiliary = cast(list[Any], auxiliary_value)
            else:
                auxiliary = []
                call_contexts["auxiliary_post_action"] = auxiliary
            auxiliary.append(context)
        return context


def _partial_json(raw_text: str, fallback: Mapping[str, Any] | None) -> dict[str, Any]:
    parsed = first_json_object(raw_text)
    if parsed is not None:
        return parsed
    output = dict(fallback or {})
    for key in ("action", "answer", "reasoning"):
        match = re.search(rf'"{key}"\s*:\s*"([^\"]*)', raw_text, flags=re.DOTALL)
        if match:
            output[key] = match.group(1).strip()
    confidence = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', raw_text)
    if confidence:
        output["confidence"] = float(confidence.group(1))
    return output


class _OpenAIPlanner:
    """OpenAI-compatible model client satisfying ESI-Bench's model protocol."""

    def __init__(
        self,
        planner: Mapping[str, Any],
        skill: str,
        *,
        auxiliary_context_builder: Callable[[Sequence[Any], str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.model = str(planner["model"])
        self.skill = skill
        self.auxiliary_context_builder = auxiliary_context_builder
        raw_sampling: object = planner.get("sampling_parameters")
        self.sampling: dict[str, Any] = (
            dict(cast(Mapping[str, Any], raw_sampling)) if isinstance(raw_sampling, Mapping) else {}
        )
        self.client = OpenAI(
            api_key=str(planner.get("api_key") or "not-required"),
            base_url=str(planner["endpoint"]),
            timeout=float(self.sampling.get("timeout", 300.0)),
            max_retries=int(self.sampling.get("max_retries", 2)),
        )

    @staticmethod
    def _contents(contents: Sequence[Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in contents:
            if isinstance(item, Path):
                output.append(image_part(path_data_url(item)))
            elif isinstance(item, dict):
                mapping = cast(dict[str, Any], item)
                if mapping.get("type") in {"text", "image_url"}:
                    output.append(mapping)
                else:
                    output.append(text_part(mapping))
            else:
                output.append(text_part(item))
        return output

    @staticmethod
    def _unsupported_sampling_error(exc: BaseException) -> bool:
        """Recognize a provider's explicit pre-generation sampling rejection."""

        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        parameter = "temperature" in text or "top_p" in text
        rejection = any(marker in text for marker in ("unsupported", "not supported", "unrecognized", "unknown"))
        return status == 400 and parameter and rejection

    def generate_json(
        self,
        contents: list[Any],
        system_instruction: str,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        fallback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, str | None]:
        del response_schema
        if system_instruction == GENERIC_JSON_INSTRUCTION:
            system = system_instruction
            routed_contents = contents
        else:
            if self.auxiliary_context_builder is None:
                raise RuntimeError("Task-specific ESI-Bench planner calls require the selected context harness.")
            system = _render_skill(self.skill, system_instruction)
            routed_contents = self.auxiliary_context_builder(contents, system_instruction)
        configured_max_tokens = int(self.sampling.get("max_completion_tokens", max_output_tokens))
        if configured_max_tokens < 1 or max_output_tokens < 1:
            raise ValueError("Planner completion-token limits must be positive.")
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._contents(routed_contents)},
            ],
            # The deployment-wide setting is a ceiling, not permission to
            # override the official runner's smaller auxiliary/final limits.
            "max_completion_tokens": min(configured_max_tokens, max_output_tokens),
            "response_format": {"type": "json_object"},
        }
        request["temperature"] = self.sampling.get("temperature", temperature)
        request["top_p"] = self.sampling.get("top_p", top_p)
        if "presence_penalty" in self.sampling:
            request["presence_penalty"] = self.sampling["presence_penalty"]
        extra_body: object = self.sampling.get("extra_body")
        if isinstance(extra_body, dict):
            request["extra_body"] = cast(dict[str, Any], extra_body)
        try:
            try:
                response = cast(Any, self.client.chat.completions.create(**request))
            except Exception as exc:
                if not self._unsupported_sampling_error(exc):
                    raise
                request.pop("temperature", None)
                request.pop("top_p", None)
                response = cast(Any, self.client.chat.completions.create(**request))
        except Exception as exc:
            raise PlannerRequestError(f"Frozen planner request failed: {exc}") from exc
        choice = response.choices[0]
        raw_text = strip_thinking(choice.message.content or "")
        return (
            _partial_json(raw_text, fallback),
            raw_text,
            str(getattr(choice, "finish_reason", None)),
        )


def _load_official_pipeline(root: Path, behavior_root: Path) -> Any:
    """Import the pinned official runner only inside the simulator process."""

    source_errors = check_upstream_source(root)
    if source_errors:
        raise RuntimeError("Unsupported ESI-Bench checkout: " + "; ".join(source_errors))
    behavior_errors = [*check_behavior_source(behavior_root), *check_omnigibson_install(behavior_root)]
    if behavior_errors:
        raise RuntimeError("Unsupported BEHAVIOR/OmniGibson environment: " + "; ".join(behavior_errors))
    active_root = root / "src" / "active_explore"
    pipeline_path = active_root / "pipeline.py"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Official ESI-Bench pipeline not found: {pipeline_path}")
    sys.path.insert(0, str(active_root))
    import importlib

    pipeline = importlib.import_module("pipeline")
    loaded = Path(str(pipeline.__file__)).resolve()
    if loaded != pipeline_path.resolve():
        raise RuntimeError(f"Imported the wrong ESI-Bench pipeline: {loaded}")
    return pipeline


def _defer_simulator_shutdown(pipeline: Any) -> Callable[[], None]:
    """Keep ``run_one`` alive until the worker response is durable.

    The pinned official runner invokes ``og.shutdown()`` in ``run_one``'s
    ``finally`` block. OmniGibson can terminate the interpreter from that call,
    before this worker has converted the official result into SHAPER records.
    Each worker handles exactly one episode, so deferring shutdown until after
    the response file is atomically persisted preserves the official episode
    lifecycle without reusing simulator state.
    """

    global _deferred_simulator_shutdown

    omnigibson = getattr(pipeline, "og", None)
    shutdown = getattr(omnigibson, "shutdown", None)
    if not callable(shutdown):
        return lambda: None

    def restore() -> None:
        setattr(omnigibson, "shutdown", shutdown)

    def deferred_shutdown() -> Any:
        restore()
        if getattr(omnigibson, "app", None) is not None:
            return shutdown()
        return None

    setattr(omnigibson, "shutdown", lambda: None)
    _deferred_simulator_shutdown = deferred_shutdown
    return restore


def _write_response(path: Path, response: Mapping[str, Any]) -> None:
    """Atomically persist a complete worker response before simulator exit."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(response, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finish_deferred_simulator_shutdown() -> None:
    """Shut down OmniGibson only after the response is safe to consume."""

    global _deferred_simulator_shutdown

    shutdown = _deferred_simulator_shutdown
    _deferred_simulator_shutdown = None
    if shutdown is None:
        return
    try:
        shutdown()
    except BaseException:
        # Some Isaac/Kit releases terminate the process from shutdown. If they
        # instead raise, the already-durable response remains authoritative.
        logger.exception("Deferred OmniGibson shutdown raised after response persistence")


def _is_environment_failure(exc: BaseException) -> bool:
    if isinstance(exc, (PlannerRequestError, HarnessBridgeError)):
        return False
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    text = " | ".join(messages).lower()
    markers = (
        "omnigibson",
        "isaac",
        "physx",
        "carb",
        "cuda",
        "vulkan",
        "egl",
        "renderer",
        "render product",
        "failed to load usd",
        "failed to load scene",
        "failed to initialize simulation",
        "failed to create render product",
        "segmentation fault",
    )
    return any(marker in text for marker in markers)


def _failure_payload(exc: BaseException) -> tuple[str, bool, str]:
    """Classify benchmark failure without turning adapter bugs into zero reward."""

    if isinstance(exc, PlannerRequestError):
        return "planner_failure", False, "planner"
    if isinstance(exc, HarnessBridgeError):
        return "harness_failure", False, "artifact"
    if isinstance(exc, OfficialRunnerError) and _is_environment_failure(exc):
        return "environment_failure", True, "environment"
    return "adapter_or_upstream_failure", False, "infrastructure"


def _history_rounds(
    result: Mapping[str, Any],
    collector: _HarnessCollector,
    task_id: str,
) -> list[dict[str, Any]]:
    raw_history = result.get("history")
    history: list[dict[str, Any]] = []
    if isinstance(raw_history, list):
        for raw_item in cast(list[Any], raw_history):
            if not isinstance(raw_item, dict):
                raise TypeError("Official ESI-Bench history entries must be dictionaries.")
            history.append(cast(dict[str, Any], raw_item))
    image_dir = Path(str(result.get("step_image_dir", "")))
    rounds: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        snapshot = collector.snapshots.get(index, {})
        before = snapshot.get("observation_before", [])
        after: list[dict[str, Any]] = []
        if index + 1 < len(history):
            next_image = image_dir / str(history[index + 1].get("image", ""))
            if next_image.is_file():
                after = [collector.images.observation(next_image)["full_frame"]]
        for extra in _extra_paths(item):
            if extra.is_file():
                after.append(collector.images.observation(extra)["full_frame"])
        if not after and isinstance(before, list):
            after = cast(list[dict[str, Any]], list(cast(list[Any], before)))
        task_instruction = str(snapshot.get("task_instruction", ""))
        if not task_instruction:
            task_instruction = "Official ESI-Bench task prompt unavailable for this auxiliary call."
        record: dict[str, Any] = {
            "record_type": "shaper_round",
            "round_index": index,
            "task_instruction": task_instruction,
            "planner_response": "\n".join(
                part
                for part in (
                    str(item.get("raw_output") or item.get("reasoning", "")).strip(),
                    str(item.get("raw_output_post_action") or "").strip(),
                )
                if part
            ),
            "command": str(item.get("action", "")),
            "observation_before": cast(list[dict[str, Any]], before) if isinstance(before, list) else [],
            "observation_after": after,
            "context_payload": snapshot.get("context_payload"),
            "harness_input": snapshot.get("harness_input"),
            "execution_steps": 1,
            "action_result": {
                **sanitized_action_result(item.get("action_result")),
                "action_valid": not bool(item.get("paper_invalid", False)),
                "reprompted": bool(item.get("paper_reprompted", False)),
                "finish_reason": str(item.get("finish_reason", "")),
            },
            "runtime_errors": [],
        }
        rounds.append(record)
    logger.info("Built %d observable rounds for %s", len(rounds), task_id)
    return rounds


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one official question and return only non-privileged SHAPER records."""

    root = Path(str(request["esi_bench_root"])).expanduser().resolve()
    behavior_root = Path(str(request["behavior_root"])).expanduser().resolve()
    run_dir = Path(str(request["run_dir"])).expanduser().resolve()
    task = cast(dict[str, Any], request["task"])
    planner = cast(dict[str, Any], request["planner"])
    runtime = cast(dict[str, Any], request["runtime"])
    skill = str(request["skill"])
    bridge = cast(dict[str, Any], request["harness_bridge"])
    task_id = str(task.get("task_id", "esi/unknown"))
    question_row = load_question_row(
        Path(str(request["questions_jsonl"])).expanduser().resolve(),
        task.get("question_id"),
    )
    expected_runner_task = str(task.get("runner_task", "")).strip()
    if str(question_row.get("runner_task", "")).strip() != expected_runner_task:
        raise ValueError(f"ESI-Bench runner_task mismatch for {task_id}.")
    question_path = resolve_canonical_question(
        root / "dataset" / "json_clean",
        task.get("question_relpath"),
        task.get("question_id"),
    )
    canonical_value: object = json.loads(question_path.read_text(encoding="utf-8"))
    if not isinstance(canonical_value, dict):
        raise ValueError(f"Canonical ESI-Bench question must be an object: {question_path}")
    canonical_row = cast(dict[str, Any], canonical_value)
    if str(canonical_row.get("runner_task", "")).strip() != expected_runner_task:
        raise ValueError(f"Canonical ESI-Bench runner_task mismatch for {task_id}.")

    skill_errors = validate_skill(skill)
    if skill_errors:
        raise ValueError("Invalid ESI-Bench skill: " + "; ".join(skill_errors))
    build_context = HarnessBridgeClient(
        Path(str(bridge["socket_path"])),
        str(bridge["token"]),
        timeout_seconds=float(bridge["timeout_seconds"]),
        max_response_bytes=int(bridge["max_response_bytes"]),
    )
    collector = _HarnessCollector(
        skill,
        build_context,
        max_steps=int(runtime["max_steps"]),
    )
    model = _OpenAIPlanner(
        planner,
        skill,
        auxiliary_context_builder=collector.route_auxiliary,
    )
    pipeline = _load_official_pipeline(root, behavior_root)
    restore_shutdown = _defer_simulator_shutdown(pipeline)
    original_build_model = pipeline.build_model_client
    original_collect = pipeline.collect_contents

    def use_frozen_planner(provider: object, api_key: object, model_name: object) -> _OpenAIPlanner:
        del provider, api_key, model_name
        return model

    pipeline.build_model_client = use_frozen_planner
    pipeline.collect_contents = collector
    try:
        config = pipeline.ActiveExploreConfig(
            task=str(task["runner_task"]),
            metadata=question_path,
            question_index=0,
            json_root=None,
            results_root=run_dir / "official_results",
            step_image_root=run_dir / "official_steps",
            provider="gpt",
            model=str(planner["model"]),
            api_key=None,
            max_steps=int(runtime["max_steps"]),
            min_steps=int(runtime["min_steps"]),
            threshold=float(runtime["confidence_threshold"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            temperature=float(runtime["temperature"]),
            top_p=float(runtime["top_p"]),
            robot=str(runtime["robot"]),
            overwrite=True,
        )
        try:
            result = cast(dict[str, Any], pipeline.run_one(config))
        except (PlannerRequestError, HarnessBridgeError):
            raise
        except BaseException as exc:
            raise OfficialRunnerError(f"Official ESI-Bench run_one failed: {exc}") from exc
    finally:
        pipeline.build_model_client = original_build_model
        pipeline.collect_contents = original_collect
        restore_shutdown()

    skipped = bool(result.get("skipped"))
    if skipped:
        skip_reason = str(result.get("skip_reason") or "official runner skipped the question")
        return {
            "ok": False,
            "environment_invalid": True,
            "termination_reason": "official_skip",
            "error": skip_reason,
        }
    correct = result.get("correct") is True
    raw_result_history: object = result.get("history")
    final_answer_value: object = result.get("final_answer")
    final_answer = cast(dict[str, Any], final_answer_value) if isinstance(final_answer_value, dict) else {}
    task_contract = ""
    if collector.snapshots:
        final_snapshot = collector.snapshots[max(collector.snapshots)]
        task_contract = str(final_snapshot.get("task_instruction", ""))
    metadata: dict[str, Any] = {
        "record_type": "shaper_episode",
        "environment_invalid": False,
        "termination_reason": str(final_answer.get("stopped_by", "completed")),
        "runtime_errors": [],
        "extra": {
            "task_id": task_id,
            "question_id": str(result.get("question_id", task.get("question_id", ""))),
            "official_steps": len(cast(list[Any], raw_result_history)) if isinstance(raw_result_history, list) else 0,
            "task_family": str(question_row.get("big_task", result.get("task_type", ""))),
            "task_subfamily": str(question_row.get("small_task", "")),
            "runner_task": expected_runner_task,
            "scene": str(result.get("scene", question_row.get("scene", ""))),
            "room": str(result.get("room", question_row.get("room", ""))),
            "question": str(result.get("question", question_row.get("question", ""))),
            "options": _decoded_json_field(question_row, "options_json"),
            "task_contract": task_contract,
            "final_answer": final_answer,
            "ground_truth": result.get("ground_truth", _decoded_json_field(question_row, "answer")),
            "official_correct": correct,
            "planner_skill": skill,
            "reference_images": collector.reference_images,
        },
    }
    return {
        "ok": True,
        "reward": 1.0 if correct else 0.0,
        "rounds": _history_rounds(result, collector, task_id),
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-path", type=Path, required=True)
    args = parser.parse_args(argv)
    response_path = args.response_path.expanduser().resolve()
    response_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = sys.stdin.read()
        request: object = json.loads(raw)
        if not isinstance(request, dict):
            raise TypeError("Worker request must be a JSON object.")
        response = run_request(cast(dict[str, Any], request))
        exit_code = 0
    except BaseException as exc:
        termination_reason, environment_invalid, failure_kind = _failure_payload(exc)
        response = {
            "ok": False,
            "environment_invalid": environment_invalid,
            "failure_kind": failure_kind,
            "termination_reason": termination_reason,
            "error": f"{type(exc).__name__}: {exc}",
        }
        logger.exception("ESI-Bench worker failed")
        exit_code = 2
    _write_response(response_path, response)
    _finish_deferred_simulator_shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
