# Copyright (c) Microsoft. All rights reserved.

"""GEPA prompt optimization for the HotPotQA agent.

Supports Azure OpenAI (Entra ID or API key) and plain OpenAI. The backend is
selected via ``--provider`` or the ``LLM_PROVIDER`` env var — see
``llm_backend.py`` for details.

Usage::

    # Azure Entra ID (default):
    az login
    python hotpotqa_gepa.py

    # Azure API key:
    python hotpotqa_gepa.py --provider azure_key

    # OpenAI:
    python hotpotqa_gepa.py --provider openai

    # With W&B experiment tracking:
    python hotpotqa_gepa.py --wandb --wandb-project gepa-hotpotqa
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple, cast

from hotpotqa_agent import (
    HotPotQATask,
    hotpotqa_agent,
    load_hotpotqa_holdout_tasks,
    load_hotpotqa_tasks,
    prompt_template_baseline,
)
from llm_backend import VALID_PROVIDERS, LLMProvider, build_reflection_config, get_provider

from agentlightning import Trainer, setup_logging
from agentlightning.algorithm.gepa import GEPA, GEPAConfig
from agentlightning.reward import find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Dataset, PromptTemplate

logger = logging.getLogger(__name__)

CandidateSelectionStrategy = Literal["pareto", "current_best"]


@dataclass(frozen=True)
class GEPAExperimentConfig:
    max_metric_calls: int = 250
    reflection_minibatch_size: int = 8
    candidate_selection_strategy: CandidateSelectionStrategy = "pareto"
    seed: int = 42
    n_runners: int = 8
    display_progress_bar: bool = True


@dataclass(frozen=True)
class PromptEvaluationResult:
    mean_reward: float
    rewards: list[float]

    @property
    def num_examples(self) -> int:
        return len(self.rewards)


@dataclass(frozen=True)
class GEPAExperimentResult:
    config: GEPAExperimentConfig
    best_prompt_template: str
    train_size: int
    dev_size: int
    holdout_size: int
    train_seed: int
    eval_seed: int
    holdout_eval_seed: int
    holdout_split: str
    inner_val_mean_reward: float
    holdout_mean_reward: float
    runtime_seconds: float
    artifact_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_setup_logging() -> None:
    if getattr(ensure_setup_logging, "_configured", False):
        return
    setup_logging()
    setattr(ensure_setup_logging, "_configured", True)


def load_train_val_dataset(
    train_size: int,
    dev_size: int,
    train_seed: int,
    eval_seed: int,
) -> Tuple[Dataset[HotPotQATask], Dataset[HotPotQATask]]:
    dataset_train, dataset_val = load_hotpotqa_tasks(
        train_size=train_size,
        dev_size=dev_size,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    return dataset_train, dataset_val


def setup_gepa_logger(file_path: str = "gepa.log") -> None:
    """Dump a copy of all the logs produced by the GEPA algorithm to a file."""

    target = Path(file_path).resolve()
    gepa_logger = logging.getLogger("agentlightning.algorithm.gepa")
    for handler in gepa_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == target:
                    return
            except Exception:
                continue

    target.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    gepa_logger.addHandler(file_handler)


async def _evaluate_prompt_template_async(
    prompt_template: PromptTemplate,
    dataset: Sequence[HotPotQATask],
) -> PromptEvaluationResult:
    runner = LitAgentRunner[HotPotQATask](AgentOpsTracer())
    store = InMemoryLightningStore()
    rewards: list[float] = []

    with runner.run_context(agent=hotpotqa_agent, store=store):
        for task in dataset:
            rollout = await runner.step(task, resources={"prompt_template": prompt_template})
            spans = await store.query_spans(rollout.rollout_id)
            reward = find_final_reward(spans)
            rewards.append(float(reward or 0.0))

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return PromptEvaluationResult(mean_reward=mean_reward, rewards=rewards)


def evaluate_prompt_template(
    prompt_template: PromptTemplate,
    dataset: Sequence[HotPotQATask],
) -> PromptEvaluationResult:
    return asyncio.run(_evaluate_prompt_template_async(prompt_template, dataset))


def build_gepa(
    provider: LLMProvider,
    experiment_config: GEPAExperimentConfig,
    *,
    use_wandb: bool = False,
    wandb_project: str = "gepa-hotpotqa",
    wandb_name: str | None = None,
) -> GEPA:
    reflection_model, reflection_model_kwargs = build_reflection_config(provider)
    wandb_init_kwargs: dict[str, str] = {"project": wandb_project}
    if wandb_name:
        wandb_init_kwargs["name"] = wandb_name

    logger.info("Using LLM provider: %s (reflection model: %s)", provider, reflection_model)

    return GEPA(
        config=GEPAConfig(
            max_metric_calls=experiment_config.max_metric_calls,
            reflection_minibatch_size=experiment_config.reflection_minibatch_size,
            candidate_selection_strategy=experiment_config.candidate_selection_strategy,
            reflection_model=reflection_model,
            reflection_model_kwargs=reflection_model_kwargs,
            seed=experiment_config.seed,
            display_progress_bar=experiment_config.display_progress_bar,
            use_wandb=use_wandb,
            wandb_init_kwargs=wandb_init_kwargs,
        ),
    )


def run_gepa_experiment(
    *,
    provider: LLMProvider,
    experiment_config: GEPAExperimentConfig,
    train_size: int = 32,
    dev_size: int = 32,
    train_seed: int = 1,
    eval_seed: int = 2023,
    holdout_size: int = 32,
    holdout_eval_seed: int | None = None,
    holdout_split: Literal["dev", "test"] = "test",
    artifact_dir: str | Path | None = None,
    use_wandb: bool = False,
    wandb_project: str = "gepa-hotpotqa",
    wandb_name: str | None = None,
) -> GEPAExperimentResult:
    """Run one GEPA configuration and evaluate the resulting prompt.

    This treats GEPA as the inner optimizer. The returned holdout score is the
    outer-loop objective used by ``gepa_autoresearch.py``.
    """

    ensure_setup_logging()
    os.environ["LLM_PROVIDER"] = provider

    artifact_path: Path | None = None
    if artifact_dir is not None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        setup_gepa_logger(str(artifact_path / "gepa.log"))

    start_time = time.perf_counter()
    dataset_train, dataset_val = load_train_val_dataset(
        train_size=train_size,
        dev_size=dev_size,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )

    if holdout_eval_seed is None:
        holdout_eval_seed = eval_seed

    if holdout_size <= 0:
        holdout_dataset = dataset_val
    elif holdout_split == "test":
        holdout_dataset = load_hotpotqa_holdout_tasks(
            test_size=holdout_size,
            eval_seed=holdout_eval_seed,
            train_seed=train_seed,
        )
    else:
        _, holdout_dataset = load_hotpotqa_tasks(
            train_size=1,
            dev_size=holdout_size,
            train_seed=train_seed,
            eval_seed=holdout_eval_seed,
        )

    algo = build_gepa(
        provider=provider,
        experiment_config=experiment_config,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    trainer = Trainer(
        algorithm=algo,
        n_runners=experiment_config.n_runners,
        initial_resources={
            "prompt_template": prompt_template_baseline(),
        },
    )
    trainer.fit(agent=hotpotqa_agent, train_dataset=dataset_train, val_dataset=dataset_val)

    best_prompt = algo.get_best_prompt()

    if artifact_path is not None:
        (artifact_path / "best_prompt.txt").write_text(best_prompt.template, encoding="utf-8")

    inner_eval = evaluate_prompt_template(best_prompt, [dataset_val[i] for i in range(len(dataset_val))])
    holdout_eval = evaluate_prompt_template(best_prompt, [holdout_dataset[i] for i in range(len(holdout_dataset))])
    runtime_seconds = time.perf_counter() - start_time

    result = GEPAExperimentResult(
        config=experiment_config,
        best_prompt_template=best_prompt.template,
        train_size=train_size,
        dev_size=dev_size,
        holdout_size=holdout_size,
        train_seed=train_seed,
        eval_seed=eval_seed,
        holdout_eval_seed=holdout_eval_seed,
        holdout_split=holdout_split,
        inner_val_mean_reward=inner_eval.mean_reward,
        holdout_mean_reward=holdout_eval.mean_reward,
        runtime_seconds=runtime_seconds,
        artifact_dir=str(artifact_path) if artifact_path is not None else None,
    )

    if artifact_path is not None:
        (artifact_path / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEPA prompt optimization for the HotPotQA agent")
    parser.add_argument(
        "--provider",
        type=str,
        choices=VALID_PROVIDERS,
        default=None,
        help="LLM backend (default: LLM_PROVIDER env var or azure_entra)",
    )
    parser.add_argument("--train-size", type=int, default=32, help="Number of HotPotQA training examples")
    parser.add_argument("--dev-size", type=int, default=32, help="Number of HotPotQA validation examples")
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=32,
        help="Number of held-out examples for final evaluation of the learned prompt",
    )
    parser.add_argument(
        "--holdout-split",
        type=str,
        choices=("dev", "test"),
        default="test",
        help="Held-out split used for the final score. 'test' uses the official HotPotQA validation split.",
    )
    parser.add_argument("--train-seed", type=int, default=1, help="DSPy HotPotQA training seed")
    parser.add_argument("--eval-seed", type=int, default=2023, help="DSPy HotPotQA eval seed")
    parser.add_argument(
        "--holdout-eval-seed",
        type=int,
        default=2024,
        help="Seed for the held-out evaluation split sampling",
    )
    parser.add_argument("--max-metric-calls", type=int, default=250, help="GEPA metric-call budget")
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=8,
        help="GEPA reflection minibatch size",
    )
    parser.add_argument(
        "--candidate-selection-strategy",
        type=str,
        choices=("pareto", "current_best"),
        default="pareto",
        help="GEPA candidate selection strategy",
    )
    parser.add_argument("--seed", type=int, default=42, help="GEPA random seed")
    parser.add_argument("--n-runners", type=int, default=8, help="Number of Agent-Lightning runners")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write logs and prompt artifacts")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="gepa-hotpotqa", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    os.environ["LLM_PROVIDER"] = provider

    experiment_config = GEPAExperimentConfig(
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy=cast(CandidateSelectionStrategy, args.candidate_selection_strategy),
        seed=args.seed,
        n_runners=args.n_runners,
    )

    result = run_gepa_experiment(
        provider=provider,
        experiment_config=experiment_config,
        train_size=args.train_size,
        dev_size=args.dev_size,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        holdout_size=args.holdout_size,
        holdout_eval_seed=args.holdout_eval_seed,
        holdout_split=cast(Literal["dev", "test"], args.holdout_split),
        artifact_dir=args.artifact_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    print("\nGEPA experiment completed.")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nBest prompt found:\n{result.best_prompt_template}")


if __name__ == "__main__":
    main()
