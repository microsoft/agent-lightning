# Copyright (c) Microsoft. All rights reserved.

"""Evaluate one SHAPER artifact pair on a benchmark split."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import agentlightning as agl
from agentlightning.algorithm import Baseline
from agentlightning.contrib.shaper import EpisodeTrace, SHAPERTraceAdapter
from agentlightning.types import Dataset, PromptTemplate

from .reproduce import ReproductionBundle, execution_strategy, load_bundle_factory


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for an official simulator evaluation."""

    factory: str
    split: str
    start_index: int
    limit: int | None
    n_runners: int
    output: Path
    skill_path: Path | None
    harness_path: Path | None


def parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluate SHAPER artifacts on a benchmark split.")
    parser.add_argument("--factory", required=True, help="Benchmark factory as module:function.")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Number of episodes; omit to evaluate the rest of the split.")
    parser.add_argument("--n-runners", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/shaper/evaluation.json"))
    parser.add_argument("--skill-path", type=Path, help="Skill artifact; defaults to the benchmark seed.")
    parser.add_argument("--harness-path", type=Path, help="Harness artifact; defaults to the benchmark seed.")
    args = parser.parse_args(argv)
    return EvaluationConfig(
        factory=str(args.factory),
        split=str(args.split),
        start_index=int(args.start_index),
        limit=cast(int | None, args.limit),
        n_runners=int(args.n_runners),
        output=cast(Path, args.output),
        skill_path=cast(Path | None, args.skill_path),
        harness_path=cast(Path | None, args.harness_path),
    )


def select_tasks(dataset: Dataset[Any], *, start_index: int, limit: int | None) -> list[Any]:
    """Select a contiguous, non-empty evaluation range."""

    if start_index < 0:
        raise ValueError("start-index must not be negative.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided.")
    if start_index >= len(dataset):
        raise IndexError(f"start-index {start_index} is outside a dataset of size {len(dataset)}.")
    stop = len(dataset) if limit is None else min(len(dataset), start_index + limit)
    return [dataset[index] for index in range(start_index, stop)]


def _resource_override(path: Path | None, fallback: PromptTemplate) -> PromptTemplate:
    if path is None:
        return fallback
    resolved = path.expanduser().resolve()
    return PromptTemplate(template=resolved.read_text(encoding="utf-8"), engine="f-string")


def prepare_resources(
    bundle: ReproductionBundle[Any],
    *,
    skill_path: Path | None,
    harness_path: Path | None,
) -> dict[str, Any]:
    """Load and validate the artifact pair used for evaluation."""

    resources = dict(bundle.initial_resources)
    seed_skill = resources.get(bundle.skill_resource_name)
    seed_harness = resources.get(bundle.harness_resource_name)
    if not isinstance(seed_skill, PromptTemplate) or not isinstance(seed_harness, PromptTemplate):
        raise TypeError("Benchmark factories must provide PromptTemplate skill and harness resources.")
    skill = _resource_override(skill_path, seed_skill)
    harness = _resource_override(harness_path, seed_harness)
    skill_errors = list(bundle.skill_validator(skill.template))
    if skill_errors:
        raise ValueError("Skill failed benchmark validation: " + "; ".join(skill_errors))
    harness_result = bundle.harness_validator.validate(harness.template)
    if not harness_result.valid:
        raise ValueError("Harness failed benchmark validation: " + "; ".join(harness_result.errors))
    resources[bundle.skill_resource_name] = skill
    resources[bundle.harness_resource_name] = harness
    return resources


def _safe_task_identity(task: object) -> dict[str, Any]:
    """Return identifiers only, never simulator configuration or scorer labels."""

    if not isinstance(task, Mapping):
        return {"type": type(task).__name__}
    allowed = ("task_id", "task_name", "question_id", "runner_task", "episode_index", "max_steps")
    return {key: task[key] for key in allowed if key in task}


async def _collect_episodes(trainer: agl.Trainer, expected: int) -> list[dict[str, Any]]:
    rollouts = list(await trainer.store.query_rollouts(sort_by="start_time", sort_order="asc"))
    if len(rollouts) != expected:
        raise RuntimeError(f"Evaluation expected {expected} rollouts, found {len(rollouts)}.")

    episodes: list[dict[str, Any]] = []
    for rollout in rollouts:
        spans = await trainer.store.query_spans(rollout_id=rollout.rollout_id, attempt_id="latest")
        trace: EpisodeTrace = (
            SHAPERTraceAdapter()
            .adapt(spans)
            .model_copy(update={"rollout_id": rollout.rollout_id, "task": None, "status": rollout.status})
        )
        episodes.append(
            {
                "rollout_id": rollout.rollout_id,
                "task": _safe_task_identity(rollout.input),
                "status": rollout.status,
                "reward": trace.final_reward,
                "environment_invalid": trace.metadata.environment_invalid,
                "termination_reason": trace.metadata.termination_reason,
                "round_count": len(trace.rounds),
                "runtime_errors": [
                    *trace.metadata.runtime_errors,
                    *(error for record in trace.rounds for error in record.runtime_errors),
                ],
                "adapter_errors": trace.adapter_errors,
            }
        )
    return episodes


def _summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(item["reward"]) for item in episodes if isinstance(item.get("reward"), (int, float))]
    invalid_count = sum(bool(item.get("environment_invalid")) for item in episodes)
    failed_count = sum(item.get("status") != "succeeded" for item in episodes)
    adapter_error_count = sum(bool(item.get("adapter_errors")) for item in episodes)
    evaluation_valid = (
        len(rewards) == len(episodes) and invalid_count == 0 and failed_count == 0 and adapter_error_count == 0
    )
    reward_sum = sum(rewards)
    return {
        "episode_count": len(episodes),
        "scored_episodes": len(rewards),
        "reward_sum": reward_sum,
        "mean_reward": reward_sum / len(rewards) if rewards else None,
        "environment_invalid_episodes": invalid_count,
        "failed_rollouts": failed_count,
        "adapter_error_episodes": adapter_error_count,
        "evaluation_valid": evaluation_valid,
    }


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run official benchmark rollouts and persist aggregate and per-episode results."""

    if config.n_runners < 1:
        raise ValueError("n-runners must be at least 1.")
    bundle = load_bundle_factory(config.factory)()
    dataset = bundle.train_dataset if config.split == "train" else bundle.val_dataset
    tasks = select_tasks(dataset, start_index=config.start_index, limit=config.limit)
    resources = prepare_resources(
        bundle,
        skill_path=config.skill_path,
        harness_path=config.harness_path,
    )
    trainer = agl.Trainer(
        algorithm=Baseline(
            polling_interval=0.05,
            max_queue_length=max(1, config.n_runners * 2),
            span_verbosity="none",
        ),
        adapter=SHAPERTraceAdapter(),
        initial_resources=resources,
        n_runners=config.n_runners,
        strategy=execution_strategy(config.n_runners),
        tracer=agl.OtelTracer(),
    )
    trainer.dev(agent=bundle.agent, train_dataset=tasks)
    episodes = asyncio.run(_collect_episodes(trainer, len(tasks)))
    report = {
        "factory": config.factory,
        "split": config.split,
        "start_index": config.start_index,
        "skill_path": str(config.skill_path.expanduser().resolve()) if config.skill_path else None,
        "harness_path": str(config.harness_path.expanduser().resolve()) if config.harness_path else None,
        "summary": _summarize(episodes),
        "episodes": episodes,
    }
    output = config.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = run_evaluation(parse_args(argv))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["evaluation_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
