# Copyright (c) Microsoft. All rights reserved.

"""Shared helpers for SHAPER's embodied benchmark integrations."""

from __future__ import annotations

import ast
import base64
import importlib
import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, cast

from openai import OpenAI

if TYPE_CHECKING:
    from agentlightning.types import LLM, NamedResources, Rollout


def require_prompt(resources: NamedResources, name: str) -> str:
    """Return one prompt resource as unformatted artifact text."""

    from agentlightning.types import PromptTemplate

    resource = resources.get(name)
    if not isinstance(resource, PromptTemplate):
        raise TypeError(f"Resource {name!r} must be a PromptTemplate.")
    return resource.template


def require_llm(resources: NamedResources, name: str, rollout: Rollout) -> LLM:
    """Return the concrete planner LLM resource for one rollout."""

    from agentlightning.types import LLM, AttemptedRollout, ProxyLLM

    resource = resources.get(name)
    if not isinstance(resource, LLM):
        raise TypeError(f"Resource {name!r} must be an LLM.")
    if isinstance(resource, ProxyLLM):
        if not isinstance(rollout, AttemptedRollout):
            raise ValueError("A ProxyLLM requires an AttemptedRollout before planner use.")
        return resource.with_attempted_rollout(rollout)
    return resource


def openai_client(resource: LLM) -> OpenAI:
    """Build a synchronous OpenAI-compatible client from an AGL resource."""

    return OpenAI(
        api_key=resource.api_key or "not-required",
        base_url=resource.get_base_url(),
        timeout=float(resource.sampling_parameters.get("timeout", 300.0)),
        max_retries=int(resource.sampling_parameters.get("max_retries", 2)),
    )


def image_data_url(image: Any, *, format_name: str = "PNG") -> str:
    """Encode an RGB simulator array with VLABench's OpenCV dependency."""

    cv2 = cast(Any, importlib.import_module("cv2"))
    np = cast(Any, importlib.import_module("numpy"))
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)

    normalized_format = format_name.strip().lower()
    extension = ".jpg" if normalized_format in {"jpg", "jpeg"} else ".png"
    ok, buffer = cv2.imencode(extension, array)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode the simulator image as {extension}.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    mime = "jpeg" if extension == ".jpg" else "png"
    return f"data:image/{mime};base64,{encoded}"


def path_data_url(path: Path) -> str:
    """Encode one official RGB path as an OpenAI image content URL."""

    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_part(url: str) -> dict[str, Any]:
    """Construct one OpenAI-compatible image content part."""

    return {"type": "image_url", "image_url": {"url": url}}


def text_part(text: object) -> dict[str, Any]:
    """Construct one OpenAI-compatible text content part."""

    return {"type": "text", "text": str(text)}


def normalize_content(value: Any) -> list[dict[str, Any]]:
    """Normalize a harness result into OpenAI-compatible content parts."""

    if isinstance(value, str):
        return [text_part(value)]
    if not isinstance(value, list):
        raise TypeError("Harness output must be a string or a list of content parts.")
    output: list[dict[str, Any]] = []
    for item in cast(list[Any], value):
        if not isinstance(item, dict):
            raise TypeError("Every multimodal harness item must be a dictionary.")
        part = cast(dict[str, Any], item)
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            output.append(part)
        elif part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
            image_url = cast(dict[str, Any], part["image_url"])
            if not isinstance(image_url.get("url"), str):
                raise TypeError("image_url.url must be a string.")
            output.append(part)
        else:
            raise TypeError("Harness content parts must be text or image_url blocks.")
    return output


def validate_multimodal_harness_output(value: Any) -> list[str]:
    """Validate the exact observable content shape accepted by both recipes."""

    if isinstance(value, str):
        return []
    if not isinstance(value, list):
        return ["Harness output must be a string or a list of content parts."]
    errors: list[str] = []
    for index, raw_item in enumerate(cast(list[Any], value)):
        if not isinstance(raw_item, dict):
            errors.append(f"Content part {index} must be a dictionary.")
            continue
        item = cast(dict[str, Any], raw_item)
        part_type = item.get("type")
        if part_type == "text":
            if set(item) != {"type", "text"} or not isinstance(item.get("text"), str):
                errors.append(f"Text part {index} must contain only string fields type and text.")
        elif part_type == "image_url":
            image_url = item.get("image_url")
            if set(item) != {"type", "image_url"} or not isinstance(image_url, dict):
                errors.append(f"Image part {index} must contain only type and image_url.")
                continue
            image_value = cast(dict[str, Any], image_url)
            url = image_value.get("url")
            if set(image_value) != {"url"} or not isinstance(url, str):
                errors.append(f"Image part {index} must contain exactly one string image_url.url.")
            elif not url.startswith("data:image/"):
                errors.append(f"Image part {index} must reuse an inline data:image URL.")
        else:
            errors.append(f"Content part {index} has unsupported type {part_type!r}.")
    return errors


def strip_thinking(text: str) -> str:
    """Remove provider-specific hidden-thinking tags from visible planner output."""

    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def first_json_object(text: str) -> dict[str, Any] | None:
    """Recover the first JSON object from an otherwise noisy completion."""

    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    return None


def completion_text(response: Any) -> tuple[str, str | None]:
    """Extract visible content and finish reason from a chat completion."""

    choice = response.choices[0]
    content = choice.message.content
    return (content if isinstance(content, str) else ""), str(getattr(choice, "finish_reason", ""))


def sanitized_action_result(value: Any) -> dict[str, Any]:
    """Keep only official, visibly returned action-result fields."""

    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[str, Any], value)
    allowed = {
        "handled",
        "operation",
        "action",
        "success",
        "error",
        "reason",
        "object",
        "target",
        "container",
        "physical_state",
        "attempts",
        "current_stack",
    }
    return {key: mapping[key] for key in allowed if key in mapping}


def load_text(directory: Path, name: str) -> str:
    """Read a UTF-8 recipe asset."""

    return (directory / name).read_text(encoding="utf-8")


def ensure_jsonable(parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Detach content parts before crossing the harness subprocess boundary."""

    return cast(list[dict[str, Any]], json.loads(json.dumps(list(parts))))


def git_revision(path: Path) -> str | None:
    """Return the containing Git checkout revision without changing files."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def git_tracked_changes(path: Path) -> tuple[str, ...] | None:
    """Return tracked files changed from HEAD, or ``None`` outside Git."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "diff", "--name-only", "HEAD", "--"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def git_head_file(path: Path, relative_path: str) -> str | None:
    """Read one UTF-8 file exactly as recorded by the checkout's HEAD."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "show", f"HEAD:{relative_path}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    return completed.stdout


def git_gitlink_revision(path: Path, relative_path: str) -> str | None:
    """Return the commit recorded for one Git submodule without initializing it."""

    try:
        checkout = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        completed = subprocess.run(
            ["git", "-C", checkout, "ls-tree", "HEAD", "--", relative_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    fields = completed.stdout.strip().split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        return None
    return fields[2]


def check_python_api(
    path: Path,
    *,
    functions: Mapping[str, Iterable[str]] | None = None,
    annotated_classes: Mapping[str, Iterable[str]] | None = None,
    class_methods: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
) -> list[str]:
    """Check a pinned upstream Python API without importing heavy runtimes."""

    if not path.is_file():
        return [f"Missing upstream Python module: {path}"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"Cannot parse upstream Python module {path}: {exc}"]

    top_level_functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    top_level_classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    errors: list[str] = []
    for name, required_parameters in (functions or {}).items():
        node = top_level_functions.get(name)
        if node is None:
            errors.append(f"Upstream API {path}:{name} is missing.")
            continue
        parameters = {argument.arg for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
        missing = sorted(set(required_parameters) - parameters)
        if missing:
            errors.append(f"Upstream API {path}:{name} is missing parameters: {', '.join(missing)}")

    for name, required_fields in (annotated_classes or {}).items():
        node = top_level_classes.get(name)
        if node is None:
            errors.append(f"Upstream API class {path}:{name} is missing.")
            continue
        fields = {
            child.target.id
            for child in node.body
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
        }
        missing = sorted(set(required_fields) - fields)
        if missing:
            errors.append(f"Upstream API class {path}:{name} is missing fields: {', '.join(missing)}")

    for class_name, required_methods in (class_methods or {}).items():
        class_node = top_level_classes.get(class_name)
        if class_node is None:
            errors.append(f"Upstream API class {path}:{class_name} is missing.")
            continue
        methods = {
            node.name: node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for method_name, required_parameters in required_methods.items():
            method = methods.get(method_name)
            if method is None:
                errors.append(f"Upstream API {path}:{class_name}.{method_name} is missing.")
                continue
            parameters = {
                argument.arg
                for argument in [
                    *method.args.posonlyargs,
                    *method.args.args,
                    *method.args.kwonlyargs,
                ]
            }
            missing = sorted(set(required_parameters) - parameters)
            if missing:
                errors.append(
                    f"Upstream API {path}:{class_name}.{method_name} is missing parameters: " + ", ".join(missing)
                )
    return errors
