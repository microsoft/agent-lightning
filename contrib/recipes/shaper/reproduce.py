# Copyright (c) Microsoft. All rights reserved.

"""Run SHAPER against an Agent Lightning benchmark bundle.

Usage from the repository root:

    OPENAI_API_KEY=... python -m contrib.recipes.shaper.reproduce \
        --factory contrib.recipes.shaper.vlabench.factory:build_bundle \
        --model qwen3.6-27b \
        --output-dir outputs/shaper

Complete factories are included for VLABench and ESI-Bench; external adapters
may provide the same :class:`ReproductionBundle` contract. This module never
stores API keys in its output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar, cast

from openai import AsyncOpenAI

import agentlightning as agl
from agentlightning.contrib.shaper import (
    DEFAULT_HARNESS_CONTRACT,
    SHAPER,
    PythonHarnessValidator,
    SHAPERRoleProtocol,
    SHAPERTraceAdapter,
    SkillValidator,
    validate_nonempty_skill,
)
from agentlightning.litagent import LitAgent
from agentlightning.types import LLM, Dataset, NamedResources, ProxyLLM

T_task = TypeVar("T_task")


@dataclass(frozen=True)
class ReproductionBundle(Generic[T_task]):
    """Benchmark objects required by the generic reproduction runner.

    ``planner_resource_name`` identifies the actual LLM resource consumed by
    the embodied planner. The optimizer reuses that resource's model and
    endpoint; factory-declared strings are not trusted as evidence of model
    identity.
    """

    agent: LitAgent[T_task]
    train_dataset: Dataset[T_task]
    val_dataset: Dataset[T_task]
    initial_resources: NamedResources
    planner_resource_name: str = "planner_llm"
    skill_resource_name: str = "skill"
    harness_resource_name: str = "harness"
    harness_contract: str = DEFAULT_HARNESS_CONTRACT
    skill_validator: SkillValidator = validate_nonempty_skill
    harness_validator: PythonHarnessValidator = field(default_factory=PythonHarnessValidator)
    judger_prompt: str | None = None
    summarizer_prompt: str | None = None
    skill_optimizer_prompt: str | None = None
    harness_optimizer_prompt: str | None = None
    role_protocol: SHAPERRoleProtocol | None = None
    provenance: Mapping[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))


@dataclass(frozen=True)
class ReproductionConfig:
    """Typed command-line configuration with the SHAPER recipe defaults."""

    factory: str
    model: str | None
    output_dir: Path
    base_url: str | None
    api_key_env: str
    n_runners: int
    validation_size: int | None
    gradient_batch_size: int
    beam_width: int
    branch_factor: int
    skill_rounds: int
    harness_rounds: int
    role_max_completion_tokens: int | None
    optimizer_temperature: float
    rollout_batch_timeout: float
    artifact_repair_attempts: int
    random_seed: int


BundleFactory = Callable[[], ReproductionBundle[Any]]

_SENSITIVE_PROVENANCE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


def validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe provenance object and reject likely credentials.

    Reproduction metadata is intentionally public-facing. Factories must record
    immutable source and protocol identifiers, never endpoint credentials.
    """

    def reject_sensitive_keys(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            for raw_key, child in mapping.items():
                if not isinstance(raw_key, str):
                    raise TypeError(f"Provenance key at {path} must be a string.")
                normalized = raw_key.lower().replace("-", "_")
                if normalized in _SENSITIVE_PROVENANCE_KEYS or normalized.endswith(
                    ("_api_key", "_authorization", "_credential", "_credentials", "_password", "_secret")
                ):
                    raise ValueError(f"Sensitive provenance key is not allowed: {path}.{raw_key}")
                reject_sensitive_keys(child, f"{path}.{raw_key}")
        elif isinstance(item, (list, tuple)):
            sequence = cast(Sequence[object], item)
            for index, child in enumerate(sequence):
                reject_sensitive_keys(child, f"{path}[{index}]")

    copied = dict(value)
    reject_sensitive_keys(copied, "provenance")
    try:
        encoded = json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Reproduction provenance must be finite JSON data: {exc}") from exc
    decoded: object = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Reproduction provenance must be a JSON object.")
    return cast(dict[str, Any], decoded)


def load_bundle_factory(spec: str) -> BundleFactory:
    """Load a ``module:function`` bundle factory without importing benchmark code here."""

    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Factory must use the form 'module:function'.")
    module = importlib.import_module(module_name)
    value: object = getattr(module, attribute_name)
    if not callable(value):
        raise TypeError(f"Factory {spec!r} is not callable.")
    return cast(BundleFactory, value)


def execution_strategy(n_runners: int) -> str | dict[str, object]:
    """Run every simulator worker in its own process."""

    if n_runners < 1:
        raise ValueError("n_runners must be at least 1.")
    return {"type": "cs", "main_process": "algorithm"}


def parse_args(argv: Sequence[str] | None = None) -> ReproductionConfig:
    """Parse CLI flags into a type-checked immutable configuration."""

    parser = argparse.ArgumentParser(description="Run two-stage SHAPER artifact evolution.")
    parser.add_argument("--factory", required=True, help="Benchmark factory as module:function.")
    parser.add_argument(
        "--model",
        help="Optional assertion for the model declared by the planner LLM resource.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shaper"))
    parser.add_argument("--base-url", help="Optional assertion for the planner endpoint declared by the factory.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--n-runners", type=int, default=1)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--skill-rounds", type=int, default=2)
    parser.add_argument("--harness-rounds", type=int, default=2)
    parser.add_argument(
        "--role-max-completion-tokens",
        type=int,
        help="Override the optimizer-role output limit declared by the planner resource.",
    )
    parser.add_argument("--optimizer-temperature", type=float, default=1.0)
    parser.add_argument(
        "--rollout-batch-timeout",
        type=float,
        default=3600.0,
        help="Wall-clock allowance per concurrent rollout wave (default: 3600s).",
    )
    parser.add_argument("--artifact-repair-attempts", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args(argv)
    return ReproductionConfig(
        factory=str(args.factory),
        model=cast(str | None, args.model),
        output_dir=cast(Path, args.output_dir),
        base_url=cast(str | None, args.base_url),
        api_key_env=str(args.api_key_env),
        n_runners=int(args.n_runners),
        validation_size=cast(int | None, args.validation_size),
        gradient_batch_size=int(args.gradient_batch_size),
        beam_width=int(args.beam_width),
        branch_factor=int(args.branch_factor),
        skill_rounds=int(args.skill_rounds),
        harness_rounds=int(args.harness_rounds),
        role_max_completion_tokens=cast(int | None, args.role_max_completion_tokens),
        optimizer_temperature=float(args.optimizer_temperature),
        rollout_batch_timeout=float(args.rollout_batch_timeout),
        artifact_repair_attempts=int(args.artifact_repair_attempts),
        random_seed=int(args.random_seed),
    )


def run_reproduction(config: ReproductionConfig) -> SHAPER[Any]:
    """Run one benchmark bundle and persist JSON-serializable optimization artifacts."""

    if config.n_runners < 1:
        raise ValueError("n_runners must be at least 1.")
    bundle = load_bundle_factory(config.factory)()
    provenance = validate_provenance(bundle.provenance)
    planner = bundle.initial_resources.get(bundle.planner_resource_name)
    if not isinstance(planner, LLM):
        raise TypeError(f"Resource {bundle.planner_resource_name!r} must be an LLM used by the embodied planner.")
    if config.model is not None and planner.model != config.model:
        raise ValueError(
            "SHAPER reproduction requires one model identity for planner and optimizer: "
            f"planner resource uses {planner.model!r}, CLI asserted {config.model!r}."
        )
    planner_endpoint = planner.get_base_url(None, None) if isinstance(planner, ProxyLLM) else planner.get_base_url()
    if config.base_url is not None and config.base_url.rstrip("/") != planner_endpoint.rstrip("/"):
        raise ValueError(
            "The optimizer endpoint cannot differ from the embodied planner endpoint: "
            f"planner uses {planner_endpoint!r}, CLI asserted {config.base_url!r}."
        )
    # Local OpenAI-compatible servers commonly do not authenticate. The SDK
    # still requires a non-empty string, while remote providers will reject the
    # placeholder clearly if the user forgot their real credential.
    api_key = os.environ.get(config.api_key_env) or planner.api_key or "not-required"

    sampling: dict[str, Any] = planner.sampling_parameters
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=planner_endpoint,
        timeout=float(sampling.get("timeout", 300.0)),
        max_retries=int(sampling.get("max_retries", 2)),
    )
    role_max_completion_tokens = config.role_max_completion_tokens or int(
        sampling.get("optimizer_max_completion_tokens", 65_536)
    )
    raw_extra_body: object = sampling.get("extra_body")
    if raw_extra_body is not None and not isinstance(raw_extra_body, dict):
        raise TypeError("planner sampling_parameters.extra_body must be a dictionary when provided.")
    role_extra_body = dict(cast(dict[str, Any], raw_extra_body)) if isinstance(raw_extra_body, dict) else {}
    for key in ("top_p", "presence_penalty"):
        if key in sampling:
            role_extra_body.setdefault(key, sampling[key])
    algorithm = SHAPER[Any](
        client,
        model=planner.model,
        skill_resource_name=bundle.skill_resource_name,
        harness_resource_name=bundle.harness_resource_name,
        validation_size=config.validation_size,
        gradient_batch_size=config.gradient_batch_size,
        beam_width=config.beam_width,
        branch_factor=config.branch_factor,
        skill_rounds=config.skill_rounds,
        harness_rounds=config.harness_rounds,
        rollout_batch_timeout=config.rollout_batch_timeout,
        optimizer_temperature=config.optimizer_temperature,
        role_max_completion_tokens=role_max_completion_tokens,
        role_extra_body=role_extra_body or None,
        artifact_repair_attempts=config.artifact_repair_attempts,
        random_seed=config.random_seed,
        harness_contract=bundle.harness_contract,
        skill_validator=bundle.skill_validator,
        harness_validator=bundle.harness_validator,
        judger_prompt=bundle.judger_prompt,
        summarizer_prompt=bundle.summarizer_prompt,
        skill_optimizer_prompt=bundle.skill_optimizer_prompt,
        harness_optimizer_prompt=bundle.harness_optimizer_prompt,
        role_protocol=bundle.role_protocol,
    )
    trainer = agl.Trainer(
        algorithm=algorithm,
        adapter=SHAPERTraceAdapter(),
        initial_resources=bundle.initial_resources,
        n_runners=config.n_runners,
        strategy=execution_strategy(config.n_runners),
        tracer=agl.OtelTracer(),
    )
    trainer.fit(
        agent=bundle.agent,
        train_dataset=bundle.train_dataset,
        val_dataset=bundle.val_dataset,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best = algorithm.get_best_candidate()
    report = {
        "configuration": {
            "factory": config.factory,
            "model": planner.model,
            "planner_resource_name": bundle.planner_resource_name,
            "n_runners": config.n_runners,
            "planner_endpoint_configured": bool(planner_endpoint),
            "validation_size": config.validation_size,
            "gradient_batch_size": config.gradient_batch_size,
            "beam_width": config.beam_width,
            "branch_factor": config.branch_factor,
            "skill_rounds": config.skill_rounds,
            "harness_rounds": config.harness_rounds,
            "role_max_completion_tokens": role_max_completion_tokens,
            "optimizer_temperature": config.optimizer_temperature,
            "rollout_batch_timeout": config.rollout_batch_timeout,
            "artifact_repair_attempts": config.artifact_repair_attempts,
            "provider_extra_body_configured": bool(role_extra_body),
            "random_seed": config.random_seed,
        },
        "provenance": provenance,
        "best_candidate": best.model_dump(mode="json"),
        "optimization_history": [event.model_dump(mode="json") for event in algorithm.get_optimization_history()],
    }
    (config.output_dir / "shaper_run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "best_skill.txt").write_text(best.skill.template.rstrip() + "\n", encoding="utf-8")
    (config.output_dir / "best_harness.py").write_text(best.harness.template.rstrip() + "\n", encoding="utf-8")
    return algorithm


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    algorithm = run_reproduction(parse_args(argv))
    best = algorithm.get_best_candidate()
    print(f"SHAPER complete: {best.version} validation_score={best.validation_score:.4f}")


if __name__ == "__main__":
    main()
