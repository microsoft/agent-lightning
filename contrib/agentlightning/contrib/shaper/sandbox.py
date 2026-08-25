# Copyright (c) Microsoft. All rights reserved.

"""Validation and subprocess execution for generated context harnesses."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, FrozenSet, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field


class HarnessValidationResult(BaseModel):
    """Outcome returned before a generated harness is admitted to a rollout."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(ge=0.0)
    output_preview: str = ""


class HarnessRuntimeError(RuntimeError):
    """Raised when an admitted harness fails in its restricted runtime."""


HarnessOutputValidator = Callable[[Any], Sequence[str]]

# Keep this list intentionally small. Optimizer-generated harnesses normally do
# not need imports; these modules are provided only for deterministic, pure
# transformations of the JSON payload supplied by the benchmark adapter.
SUPPORTED_HARNESS_IMPORTS: FrozenSet[str] = frozenset(
    {"collections", "functools", "itertools", "json", "math", "re", "textwrap"}
)


def _validate_allowed_imports(allowed_imports: FrozenSet[str]) -> None:
    unsupported = sorted(set(allowed_imports) - SUPPORTED_HARNESS_IMPORTS)
    if unsupported:
        raise ValueError(
            "Unsupported harness imports: "
            + ", ".join(unsupported)
            + ". Supported modules: "
            + ", ".join(sorted(SUPPORTED_HARNESS_IMPORTS))
            + "."
        )


@dataclass(frozen=True)
class PythonHarnessRuntime:
    """Call one harness function only through the isolated worker process.

    This class deliberately does not expose the compiled function in the parent
    process. Validation and rollout execution therefore use the same builtins,
    import policy, resource limits, and wall-clock timeout.
    """

    source: str
    function_name: str = "build_context"
    timeout_seconds: float = 2.0
    memory_limit_mb: int = 512
    max_output_chars: int = 8_000_000
    allowed_imports: FrozenSet[str] = frozenset()
    output_validator: HarnessOutputValidator | None = None

    def __post_init__(self) -> None:
        if not self.function_name.isidentifier():
            raise ValueError("function_name must be a valid Python identifier.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.memory_limit_mb < 1:
            raise ValueError("memory_limit_mb must be at least 1.")
        if self.max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1.")
        _validate_allowed_imports(self.allowed_imports)

    def __call__(self, *args: Any) -> Any:
        """Execute the harness with JSON-serializable arguments."""

        try:
            response = _run_isolated(
                source=self.source,
                function_name=self.function_name,
                args=args,
                timeout_seconds=self.timeout_seconds,
                memory_limit_mb=self.memory_limit_mb,
                max_output_chars=self.max_output_chars,
                allowed_imports=self.allowed_imports,
                include_output=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessRuntimeError(f"Harness execution exceeded {self.timeout_seconds:.2f}s timeout.") from exc
        except (TypeError, ValueError) as exc:
            raise HarnessRuntimeError(f"Harness arguments are not JSON-serializable: {exc}") from exc
        if response.get("valid") is not True:
            raw_errors: object = response.get("errors", [])
            errors = cast(list[object], raw_errors) if isinstance(raw_errors, list) else [raw_errors]
            raise HarnessRuntimeError("; ".join(str(item) for item in errors))
        output = response.get("output")
        if self.output_validator is not None:
            errors = list(self.output_validator(output))
            if errors:
                raise HarnessRuntimeError("; ".join(errors))
        return output


def _empty_history_probe() -> tuple[list[dict[str, Any]]]:
    return ([],)


def _single_history_probe() -> tuple[list[dict[str, Any]]]:
    return (
        [
            {
                "round_index": 0,
                "task_instruction": "sandbox smoke",
                "planner_response": "inspect",
                "command": "inspect target",
                "observation_before": [],
                "observation_after": [],
                "action_result": {"status": "unknown"},
                "execution_steps": 0,
                "runtime_errors": [],
            }
        ],
    )


def _multi_history_probe() -> tuple[list[dict[str, Any]]]:
    return (
        [
            {
                "round_index": 0,
                "task_instruction": "different probe",
                "planner_response": "move left",
                "command": "move left",
                "observation_before": [{"type": "text", "text": "target absent"}],
                "observation_after": [{"type": "text", "text": "target visible"}],
                "action_result": {"handled": True},
                "execution_steps": 7,
                "runtime_errors": [],
            },
            {
                "round_index": 1,
                "task_instruction": "different probe",
                "planner_response": "answer",
                "command": "answer middle",
                "observation_before": [],
                "observation_after": [],
                "action_result": {"handled": False},
                "execution_steps": 1,
                "runtime_errors": ["probe-runtime-error"],
            },
        ],
    )


@dataclass(frozen=True)
class PythonHarnessValidator:
    """Apply static checks and execute several paths in the rollout runtime.

    The worker is a reproducibility and fault-containment layer, not a hardened
    operating-system security boundary. Deploy optimizer-generated code in a
    container or VM when it is not trusted by the machine owner.
    """

    function_name: str = "build_context"
    smoke_args: Sequence[Any] = field(default_factory=_single_history_probe)
    additional_smoke_args: Sequence[Sequence[Any]] = field(
        default_factory=lambda: (_empty_history_probe(), _multi_history_probe())
    )
    timeout_seconds: float = 2.0
    max_source_chars: int = 100_000
    memory_limit_mb: int = 512
    max_output_chars: int = 8_000_000
    allowed_imports: FrozenSet[str] = frozenset()
    output_validator: HarnessOutputValidator | None = None

    _forbidden_names: ClassVar[FrozenSet[str]] = frozenset(
        {
            "__builtins__",
            "__import__",
            "breakpoint",
            "compile",
            "delattr",
            "dir",
            "eval",
            "exec",
            "exit",
            "getattr",
            "globals",
            "help",
            "input",
            "locals",
            "open",
            "quit",
            "setattr",
            "vars",
        }
    )
    _forbidden_attributes: ClassVar[FrozenSet[str]] = frozenset(
        {
            "chmod",
            "glob",
            "iterdir",
            "mkdir",
            "open",
            "popen",
            "read_bytes",
            "read_text",
            "remove",
            "rename",
            "replace",
            "resolve",
            "rglob",
            "rmdir",
            "symlink_to",
            "system",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
    )

    def __post_init__(self) -> None:
        if not self.function_name.isidentifier():
            raise ValueError("function_name must be a valid Python identifier.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_source_chars < 1:
            raise ValueError("max_source_chars must be at least 1.")
        if self.memory_limit_mb < 1:
            raise ValueError("memory_limit_mb must be at least 1.")
        if self.max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1.")
        _validate_allowed_imports(self.allowed_imports)

    def validate(self, source: str) -> HarnessValidationResult:
        """Reject unsafe source and smoke multiple execution paths."""

        started = time.monotonic()
        errors = self._static_errors(source)
        if errors:
            return HarnessValidationResult(valid=False, errors=errors, duration_seconds=time.monotonic() - started)

        previews: list[str] = []
        probes = [tuple(self.smoke_args), *(tuple(args) for args in self.additional_smoke_args)]
        for index, args in enumerate(probes):
            try:
                response = _run_isolated(
                    source=source,
                    function_name=self.function_name,
                    args=args,
                    timeout_seconds=self.timeout_seconds,
                    memory_limit_mb=self.memory_limit_mb,
                    max_output_chars=self.max_output_chars,
                    allowed_imports=self.allowed_imports,
                    include_output=self.output_validator is not None,
                )
            except (TypeError, ValueError) as exc:
                return HarnessValidationResult(
                    valid=False,
                    errors=[f"Harness smoke arguments are not JSON-serializable: {exc}"],
                    duration_seconds=time.monotonic() - started,
                )
            except subprocess.TimeoutExpired:
                return HarnessValidationResult(
                    valid=False,
                    errors=[f"Harness probe {index} exceeded {self.timeout_seconds:.2f}s timeout."],
                    duration_seconds=time.monotonic() - started,
                )
            except HarnessRuntimeError as exc:
                return HarnessValidationResult(
                    valid=False,
                    errors=[f"Probe {index}: {exc}"],
                    duration_seconds=time.monotonic() - started,
                )

            if response.get("valid") is not True:
                raw_errors: object = response.get("errors", [])
                probe_errors = cast(list[object], raw_errors) if isinstance(raw_errors, list) else [raw_errors]
                return HarnessValidationResult(
                    valid=False,
                    errors=[f"Probe {index}: {item}" for item in probe_errors],
                    duration_seconds=time.monotonic() - started,
                )
            if self.output_validator is not None:
                output_errors = list(self.output_validator(response.get("output")))
                if output_errors:
                    return HarnessValidationResult(
                        valid=False,
                        errors=[f"Probe {index}: {item}" for item in output_errors],
                        duration_seconds=time.monotonic() - started,
                    )
            previews.append(str(response.get("output_preview", "")))

        return HarnessValidationResult(
            valid=True,
            duration_seconds=time.monotonic() - started,
            output_preview=" | ".join(previews)[:500],
        )

    def runtime(self, source: str) -> PythonHarnessRuntime:
        """Validate source and return its only supported execution interface."""

        result = self.validate(source)
        if not result.valid:
            raise ValueError("Invalid context harness: " + "; ".join(result.errors))
        return PythonHarnessRuntime(
            source=source,
            function_name=self.function_name,
            timeout_seconds=self.timeout_seconds,
            memory_limit_mb=self.memory_limit_mb,
            max_output_chars=self.max_output_chars,
            allowed_imports=self.allowed_imports,
            output_validator=self.output_validator,
        )

    def _static_errors(self, source: str) -> list[str]:
        errors: list[str] = []
        if not source.strip():
            return ["Harness source is empty."]
        if len(source) > self.max_source_chars:
            return [f"Harness source exceeds {self.max_source_chars} characters."]

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return [f"Syntax error at line {exc.lineno}: {exc.msg}"]

        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        matching = [node for node in functions if node.name == self.function_name]
        if len(matching) != 1:
            errors.append(f"Harness must define exactly one top-level {self.function_name} function.")
        elif isinstance(matching[0], ast.AsyncFunctionDef):
            errors.append(f"{self.function_name} must be synchronous.")

        for node in ast.walk(tree):
            line = getattr(node, "lineno", "?")
            if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.While)):
                errors.append(f"{type(node).__name__} is not allowed (line {line}).")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = self._imported_modules(node)
                denied = sorted(module for module in imported if module not in self.allowed_imports)
                if denied:
                    errors.append(f"Import is not allowed for {', '.join(denied)} (line {line}).")
                for alias in node.names:
                    imported_name = alias.name
                    bound_name = alias.asname or imported_name.split(".", maxsplit=1)[0]
                    if any(part.startswith("_") for part in imported_name.split(".")):
                        errors.append(f"Private import {imported_name!r} is forbidden at line {line}.")
                    if bound_name.startswith("_") or bound_name in self._forbidden_names:
                        errors.append(f"Forbidden import binding {bound_name!r} at line {line}.")
            elif isinstance(node, ast.Name):
                if node.id in self._forbidden_names or node.id.startswith("__"):
                    errors.append(f"Forbidden name {node.id!r} at line {line}.")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    errors.append(f"Private attribute access is forbidden at line {line}.")
                elif node.attr in self._forbidden_attributes:
                    errors.append(f"Forbidden attribute {node.attr!r} at line {line}.")

        return list(dict.fromkeys(errors))

    @staticmethod
    def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name.split(".", maxsplit=1)[0] for alias in node.names]
        if node.module is None:
            return []
        return [node.module.split(".", maxsplit=1)[0]]


def _run_isolated(
    *,
    source: str,
    function_name: str,
    args: Sequence[Any],
    timeout_seconds: float,
    memory_limit_mb: int,
    max_output_chars: int,
    allowed_imports: FrozenSet[str],
    include_output: bool,
) -> dict[str, Any]:
    request = json.dumps(
        {
            "source": source,
            "function_name": function_name,
            "args": list(args),
            "memory_limit_mb": memory_limit_mb,
            "cpu_limit_seconds": max(1, int(timeout_seconds) + 1),
            "max_output_chars": max_output_chars,
            "allowed_imports": sorted(allowed_imports),
            "include_output": include_output,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _ISOLATED_RUNNER],
        input=request,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    try:
        decoded: object = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip()[-500:] or f"isolated process exited {completed.returncode}"
        raise HarnessRuntimeError(f"Harness worker returned no valid JSON: {detail}") from exc
    if not isinstance(decoded, dict):
        raise HarnessRuntimeError("Harness worker returned an invalid response.")
    response = cast(dict[str, Any], decoded)
    if completed.returncode != 0 and response.get("valid") is True:
        response = {"valid": False, "errors": [f"Harness worker exited {completed.returncode}."]}
    return response


_ISOLATED_RUNNER = r"""
import json
import sys
from typing import Any


def finish(payload, code=0):
    sys.stdout.write(json.dumps(payload))
    raise SystemExit(code)


try:
    request = json.loads(sys.stdin.read())
    memory_limit_mb = int(request.get("memory_limit_mb", 512))
    cpu_limit_seconds = int(request.get("cpu_limit_seconds", 2))
    try:
        import resource
        memory_bytes = memory_limit_mb * 1024 * 1024
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds))
    except (ImportError, OSError, ValueError):
        pass

    allowed_imports = set(request.get("allowed_imports", []))
    real_import = __import__

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in allowed_imports:
            raise ImportError(f"Import of {root!r} is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        "Exception": Exception,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "object": object,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    if allowed_imports:
        safe_builtins["__import__"] = limited_import

    namespace = {"__builtins__": safe_builtins, "Any": Any}
    exec(compile(request["source"], "<generated-harness>", "exec"), namespace, namespace)
    function_name = request["function_name"]
    function = namespace.get(function_name)
    if not callable(function):
        finish({"valid": False, "errors": [f"{function_name} is not callable"]}, 1)

    output = function(*request.get("args", []))
    output_json = json.dumps(output)
    max_output_chars = int(request.get("max_output_chars", 8000000))
    if len(output_json) > max_output_chars:
        finish({"valid": False, "errors": [f"Harness output exceeds {max_output_chars} characters"]}, 1)
    response = {"valid": True, "errors": [], "output_preview": output_json[:500]}
    if request.get("include_output"):
        response["output"] = output
    finish(response, 0)
except BaseException as exc:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise
    finish({"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}, 1)
"""
