# Copyright (c) Microsoft. All rights reserved.

"""Autonomous outer-loop search for GEPA hyperparameters.

This script keeps GEPA as the inner prompt optimizer and adds an autoresearch-
style outer loop that:

1. proposes a GEPA configuration,
2. runs a bounded GEPA experiment,
3. evaluates the learned prompt on a held-out split,
4. accepts or discards the proposal,
5. logs the full experiment history.

The default design mirrors the key autoresearch idea of *fixed-budget*
experiments: every trial uses the same GEPA metric-call budget unless you opt in
with ``--tune-max-metric-calls``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from hotpotqa_gepa import GEPAExperimentConfig, GEPAExperimentResult, run_gepa_experiment
from llm_backend import VALID_PROVIDERS, LLMProvider, get_provider, make_client
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CandidateSelectionStrategy = Literal["pareto", "current_best"]
ProposalPolicy = Literal["llm", "mutation", "random"]

ConfigKey = tuple[int, int, str, int, int]

DEFAULT_REFLECTION_MINIBATCH_OPTIONS = [2, 4, 8, 12, 16]
DEFAULT_N_RUNNER_OPTIONS = [1, 2, 4, 8]
DEFAULT_SEED_OPTIONS = [0, 1, 2, 7, 42, 123]
DEFAULT_MAX_METRIC_CALL_OPTIONS = [64, 96, 128, 160, 192, 256]
DEFAULT_CANDIDATE_STRATEGIES: list[CandidateSelectionStrategy] = ["pareto", "current_best"]


@dataclass(frozen=True)
class SearchSpace:
    reflection_minibatch_options: list[int]
    candidate_selection_options: list[CandidateSelectionStrategy]
    n_runner_options: list[int]
    seed_options: list[int]
    max_metric_call_options: list[int]


@dataclass(frozen=True)
class TrialSummary:
    iteration: int
    accepted: bool
    proposal_source: str
    hypothesis: str
    config: GEPAExperimentConfig
    result: GEPAExperimentResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "iteration": self.iteration,
            "accepted": self.accepted,
            "proposal_source": self.proposal_source,
            "hypothesis": self.hypothesis,
            "config": asdict(self.config),
            "error": self.error,
        }
        if self.result is not None:
            payload["result"] = self.result.to_dict()
        return payload


class ProposalConfig(BaseModel):
    max_metric_calls: int = Field(description="GEPA metric-call budget for the trial")
    reflection_minibatch_size: int = Field(description="GEPA reflection minibatch size")
    candidate_selection_strategy: CandidateSelectionStrategy = Field(description="GEPA candidate selection strategy")
    seed: int = Field(description="Random seed for GEPA")
    n_runners: int = Field(description="Number of Agent-Lightning runners to use")


class ResearchProposal(BaseModel):
    hypothesis: str = Field(description="Why this config might outperform the current best")
    config: ProposalConfig


def parse_int_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected a comma-separated list of integers")
    return [int(value) for value in values]


def ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def history_file_path(run_dir: Path) -> Path:
    return run_dir / "history.jsonl"


def best_result_path(run_dir: Path) -> Path:
    return run_dir / "best_result.json"


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    path = history_file_path(run_dir)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def append_history_entry(run_dir: Path, entry: TrialSummary) -> None:
    path = history_file_path(run_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict()) + "\n")


def write_best_result(run_dir: Path, entry: TrialSummary) -> None:
    path = best_result_path(run_dir)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entry.to_dict(), handle, indent=2)


def trial_dir(run_dir: Path, iteration: int) -> Path:
    return run_dir / f"trial_{iteration:03d}"


def config_to_key(config: GEPAExperimentConfig) -> ConfigKey:
    return (
        config.max_metric_calls,
        config.reflection_minibatch_size,
        config.candidate_selection_strategy,
        config.seed,
        config.n_runners,
    )


def history_best_entry(history: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    scored_entries = [
        entry for entry in history if entry.get("result") and entry["result"].get("holdout_mean_reward") is not None
    ]
    if not scored_entries:
        return None
    return max(scored_entries, key=lambda entry: float(entry["result"]["holdout_mean_reward"]))


def collect_seen_configs(history: Sequence[dict[str, Any]]) -> set[ConfigKey]:
    seen: set[ConfigKey] = set()
    for entry in history:
        config = entry.get("config")
        if not config:
            continue
        seen.add(
            (
                int(config["max_metric_calls"]),
                int(config["reflection_minibatch_size"]),
                str(config["candidate_selection_strategy"]),
                int(config["seed"]),
                int(config["n_runners"]),
            )
        )
    return seen


def get_research_model_name(provider: LLMProvider) -> str:
    if provider in ("azure_entra", "azure_key"):
        return (
            os.environ.get("AZURE_OPENAI_RESEARCH_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_GRADER_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        )
    return (
        os.environ.get("OPENAI_RESEARCH_MODEL")
        or os.environ.get("OPENAI_GRADER_MODEL")
        or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    )


def baseline_config(args: argparse.Namespace) -> GEPAExperimentConfig:
    return GEPAExperimentConfig(
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.initial_reflection_minibatch_size,
        candidate_selection_strategy=cast(CandidateSelectionStrategy, args.initial_candidate_selection_strategy),
        seed=args.initial_seed,
        n_runners=args.initial_n_runners,
    )


def make_search_space(args: argparse.Namespace) -> SearchSpace:
    max_metric_call_options = [args.max_metric_calls]
    if args.tune_max_metric_calls:
        max_metric_call_options = sorted(set(parse_int_list(args.max_metric_call_options)))

    return SearchSpace(
        reflection_minibatch_options=sorted(set(parse_int_list(args.reflection_minibatch_options))),
        candidate_selection_options=[
            cast(CandidateSelectionStrategy, value.strip())
            for value in args.candidate_selection_options.split(",")
            if value.strip()
        ],
        n_runner_options=sorted(set(parse_int_list(args.n_runner_options))),
        seed_options=sorted(set(parse_int_list(args.seed_options))),
        max_metric_call_options=max_metric_call_options,
    )


def clamp_to_allowed(value: int, allowed: Sequence[int]) -> int:
    return min(allowed, key=lambda option: (abs(option - value), option))


def canonicalize_config(config: GEPAExperimentConfig, space: SearchSpace, dev_size: int) -> GEPAExperimentConfig:
    minibatch = clamp_to_allowed(config.reflection_minibatch_size, space.reflection_minibatch_options)
    minibatch = max(1, min(minibatch, max(1, dev_size)))

    strategy: CandidateSelectionStrategy
    if config.candidate_selection_strategy in space.candidate_selection_options:
        strategy = config.candidate_selection_strategy
    else:
        strategy = space.candidate_selection_options[0]

    return GEPAExperimentConfig(
        max_metric_calls=clamp_to_allowed(config.max_metric_calls, space.max_metric_call_options),
        reflection_minibatch_size=minibatch,
        candidate_selection_strategy=strategy,
        seed=clamp_to_allowed(config.seed, space.seed_options),
        n_runners=clamp_to_allowed(config.n_runners, space.n_runner_options),
        display_progress_bar=True,
    )


def entry_to_short_line(entry: dict[str, Any]) -> str:
    result: dict[str, Any] = entry.get("result") or {}
    config: dict[str, Any] = entry.get("config") or {}
    holdout = result.get("holdout_mean_reward")
    inner = result.get("inner_val_mean_reward")
    return (
        f"iter={entry.get('iteration')} accepted={entry.get('accepted')} "
        f"holdout={holdout} inner={inner} "
        f"cfg={{budget={config.get('max_metric_calls')}, minibatch={config.get('reflection_minibatch_size')}, "
        f"strategy={config.get('candidate_selection_strategy')}, seed={config.get('seed')}, runners={config.get('n_runners')}}} "
        f"hypothesis={entry.get('hypothesis')}"
    )


def summarize_history(history: Sequence[dict[str, Any]], limit: int = 8) -> str:
    if not history:
        return "No prior trials yet."

    scored = [entry for entry in history if entry.get("result")]
    if not scored:
        recent = history[-limit:]
        return "\n".join(entry_to_short_line(entry) for entry in recent)

    best = sorted(scored, key=lambda entry: float(entry["result"]["holdout_mean_reward"]), reverse=True)[
        : max(1, limit // 2)
    ]
    recent = history[-max(1, limit - len(best)) :]

    lines = ["Top trials:"]
    lines.extend(f"- {entry_to_short_line(entry)}" for entry in best)
    lines.append("Recent trials:")
    lines.extend(f"- {entry_to_short_line(entry)}" for entry in recent)
    return "\n".join(lines)


def mutate_config(
    base: GEPAExperimentConfig, rng: random.Random, space: SearchSpace, dev_size: int
) -> GEPAExperimentConfig:
    proposal = {
        "max_metric_calls": base.max_metric_calls,
        "reflection_minibatch_size": base.reflection_minibatch_size,
        "candidate_selection_strategy": base.candidate_selection_strategy,
        "seed": base.seed,
        "n_runners": base.n_runners,
    }
    knobs = ["reflection_minibatch_size", "candidate_selection_strategy", "seed", "n_runners"]
    if len(space.max_metric_call_options) > 1:
        knobs.append("max_metric_calls")

    num_mutations = rng.randint(1, min(3, len(knobs)))
    for knob in rng.sample(knobs, k=num_mutations):
        if knob == "reflection_minibatch_size":
            proposal[knob] = rng.choice(space.reflection_minibatch_options)
        elif knob == "candidate_selection_strategy":
            proposal[knob] = rng.choice(space.candidate_selection_options)
        elif knob == "seed":
            proposal[knob] = rng.choice(space.seed_options)
        elif knob == "n_runners":
            proposal[knob] = rng.choice(space.n_runner_options)
        elif knob == "max_metric_calls":
            proposal[knob] = rng.choice(space.max_metric_call_options)

    return canonicalize_config(
        GEPAExperimentConfig(
            max_metric_calls=int(proposal["max_metric_calls"]),
            reflection_minibatch_size=int(proposal["reflection_minibatch_size"]),
            candidate_selection_strategy=cast(CandidateSelectionStrategy, proposal["candidate_selection_strategy"]),
            seed=int(proposal["seed"]),
            n_runners=int(proposal["n_runners"]),
        ),
        space,
        dev_size,
    )


def random_config(rng: random.Random, space: SearchSpace, dev_size: int) -> GEPAExperimentConfig:
    return canonicalize_config(
        GEPAExperimentConfig(
            max_metric_calls=rng.choice(space.max_metric_call_options),
            reflection_minibatch_size=rng.choice(space.reflection_minibatch_options),
            candidate_selection_strategy=rng.choice(space.candidate_selection_options),
            seed=rng.choice(space.seed_options),
            n_runners=rng.choice(space.n_runner_options),
        ),
        space,
        dev_size,
    )


def propose_with_llm(
    *,
    provider: LLMProvider,
    research_model: str,
    current_best: GEPAExperimentConfig,
    history: Sequence[dict[str, Any]],
    space: SearchSpace,
    dev_size: int,
) -> ResearchProposal:
    client = make_client(provider)

    search_space_description = {
        "max_metric_calls": space.max_metric_call_options,
        "reflection_minibatch_size": [
            value for value in space.reflection_minibatch_options if value <= max(1, dev_size)
        ],
        "candidate_selection_strategy": list(space.candidate_selection_options),
        "seed": space.seed_options,
        "n_runners": space.n_runner_options,
    }

    best_score = None
    best_entry = history_best_entry(history)
    if best_entry and best_entry.get("result"):
        best_score = best_entry["result"].get("holdout_mean_reward")

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are an autonomous research engineer optimizing GEPA hyperparameters. "
                "Your job is to suggest a single next experiment that is plausible, non-duplicate, and "
                "cost-aware. Prefer small, targeted changes from the current best unless history suggests a broader pivot. "
                "Always stay inside the allowed search space."
            ),
        },
        {
            "role": "user",
            "content": (
                "Current best GEPA configuration:\n"
                f"{json.dumps(asdict(current_best), indent=2)}\n\n"
                f"Current best holdout score: {best_score}\n\n"
                "Allowed search space:\n"
                f"{json.dumps(search_space_description, indent=2)}\n\n"
                "Experiment history:\n"
                f"{summarize_history(history)}\n\n"
                "Return exactly one promising next configuration and a short hypothesis."
            ),
        },
    ]

    response = client.chat.completions.parse(
        model=research_model,
        messages=messages,
        response_format=ResearchProposal,
        temperature=1.0,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Research proposal parsing returned None")

    config = canonicalize_config(
        GEPAExperimentConfig(
            max_metric_calls=parsed.config.max_metric_calls,
            reflection_minibatch_size=parsed.config.reflection_minibatch_size,
            candidate_selection_strategy=parsed.config.candidate_selection_strategy,
            seed=parsed.config.seed,
            n_runners=parsed.config.n_runners,
        ),
        space,
        dev_size,
    )
    return ResearchProposal(
        hypothesis=parsed.hypothesis,
        config=ProposalConfig(
            max_metric_calls=config.max_metric_calls,
            reflection_minibatch_size=config.reflection_minibatch_size,
            candidate_selection_strategy=config.candidate_selection_strategy,
            seed=config.seed,
            n_runners=config.n_runners,
        ),
    )


def unique_candidate(
    candidate: GEPAExperimentConfig,
    *,
    current_best: GEPAExperimentConfig,
    seen: set[ConfigKey],
    rng: random.Random,
    space: SearchSpace,
    dev_size: int,
) -> GEPAExperimentConfig:
    if config_to_key(candidate) not in seen:
        return candidate

    for _ in range(20):
        mutated = mutate_config(current_best, rng, space, dev_size)
        if config_to_key(mutated) not in seen:
            return mutated

    for _ in range(20):
        random_candidate = random_config(rng, space, dev_size)
        if config_to_key(random_candidate) not in seen:
            return random_candidate

    return candidate


def build_trial_summary(
    *,
    iteration: int,
    accepted: bool,
    proposal_source: str,
    hypothesis: str,
    config: GEPAExperimentConfig,
    result: GEPAExperimentResult | None = None,
    error: str | None = None,
) -> TrialSummary:
    return TrialSummary(
        iteration=iteration,
        accepted=accepted,
        proposal_source=proposal_source,
        hypothesis=hypothesis,
        config=config,
        result=result,
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch-like outer loop for GEPA hyperparameter tuning")
    parser.add_argument(
        "--provider",
        type=str,
        choices=VALID_PROVIDERS,
        default=None,
        help="LLM backend (default: LLM_PROVIDER env var or azure_entra)",
    )
    parser.add_argument("--iterations", type=int, default=8, help="Number of outer-loop trials to run")
    parser.add_argument(
        "--proposal-policy",
        type=str,
        choices=("llm", "mutation", "random"),
        default="llm",
        help="How to propose the next GEPA configuration",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="gepa_autoresearch_runs/default",
        help="Directory for experiment history and per-trial artifacts",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing history file if present")
    parser.add_argument("--min-improvement", type=float, default=1e-6, help="Minimum holdout improvement to accept")
    parser.add_argument("--train-size", type=int, default=32, help="Inner-loop HotPotQA train size")
    parser.add_argument("--dev-size", type=int, default=32, help="Inner-loop HotPotQA dev size")
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=32,
        help="Held-out evaluation size for the outer-loop objective",
    )
    parser.add_argument(
        "--holdout-split",
        type=str,
        choices=("dev", "test"),
        default="test",
        help="Held-out split used by the outer loop. 'test' uses the official HotPotQA validation split.",
    )
    parser.add_argument("--train-seed", type=int, default=1, help="HotPotQA train seed")
    parser.add_argument("--eval-seed", type=int, default=2023, help="HotPotQA eval seed for inner dev")
    parser.add_argument(
        "--holdout-eval-seed",
        type=int,
        default=2024,
        help="HotPotQA eval seed for held-out scoring",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=128,
        help="Default fixed GEPA metric-call budget per trial",
    )
    parser.add_argument(
        "--tune-max-metric-calls",
        action="store_true",
        help="Allow the outer loop to search over GEPA metric-call budgets as well",
    )
    parser.add_argument(
        "--max-metric-call-options",
        type=str,
        default=",".join(str(value) for value in DEFAULT_MAX_METRIC_CALL_OPTIONS),
        help="Comma-separated GEPA metric-call budgets to consider when --tune-max-metric-calls is enabled",
    )
    parser.add_argument(
        "--reflection-minibatch-options",
        type=str,
        default=",".join(str(value) for value in DEFAULT_REFLECTION_MINIBATCH_OPTIONS),
        help="Comma-separated reflection minibatch sizes to search",
    )
    parser.add_argument(
        "--candidate-selection-options",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_STRATEGIES),
        help="Comma-separated candidate selection strategies to search",
    )
    parser.add_argument(
        "--n-runner-options",
        type=str,
        default=",".join(str(value) for value in DEFAULT_N_RUNNER_OPTIONS),
        help="Comma-separated runner counts to search",
    )
    parser.add_argument(
        "--seed-options",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SEED_OPTIONS),
        help="Comma-separated GEPA seeds to search",
    )
    parser.add_argument(
        "--initial-reflection-minibatch-size",
        type=int,
        default=8,
        help="Baseline reflection minibatch size before the search starts",
    )
    parser.add_argument(
        "--initial-candidate-selection-strategy",
        type=str,
        choices=("pareto", "current_best"),
        default="pareto",
        help="Baseline candidate selection strategy before the search starts",
    )
    parser.add_argument("--initial-seed", type=int, default=42, help="Baseline GEPA seed")
    parser.add_argument("--initial-n-runners", type=int, default=8, help="Baseline runner count")
    parser.add_argument("--search-seed", type=int, default=0, help="Random seed for outer-loop mutation/random search")
    parser.add_argument(
        "--research-model",
        type=str,
        default=None,
        help="Optional override for the LLM used to propose the next GEPA configuration",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B for the inner GEPA experiments")
    parser.add_argument("--wandb-project", type=str, default="gepa-hotpotqa-autoresearch", help="W&B project name")
    parser.add_argument("--wandb-name-prefix", type=str, default="autoresearch", help="W&B run name prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    os.environ["LLM_PROVIDER"] = provider

    run_dir = Path(args.run_dir)
    ensure_log_dir(run_dir)

    history: list[dict[str, Any]]
    if args.resume:
        history = load_history(run_dir)
    else:
        history = []
        history_path = history_file_path(run_dir)
        if history_path.exists():
            history_path.unlink()
        best_path = best_result_path(run_dir)
        if best_path.exists():
            best_path.unlink()

    rng = random.Random(args.search_seed)
    space = make_search_space(args)
    research_model = args.research_model or get_research_model_name(provider)

    best_history_entry = history_best_entry(history)
    current_best_config = baseline_config(args)
    current_best_score = float("-inf")
    if best_history_entry is not None:
        config = best_history_entry["config"]
        current_best_config = GEPAExperimentConfig(
            max_metric_calls=int(config["max_metric_calls"]),
            reflection_minibatch_size=int(config["reflection_minibatch_size"]),
            candidate_selection_strategy=cast(CandidateSelectionStrategy, config["candidate_selection_strategy"]),
            seed=int(config["seed"]),
            n_runners=int(config["n_runners"]),
        )
        current_best_score = float(best_history_entry["result"]["holdout_mean_reward"])

    seen = collect_seen_configs(history)
    start_iteration = len(history)

    logger.info("Using provider=%s research_model=%s", provider, research_model)
    logger.info("Resuming with %d existing trials", start_iteration)

    for iteration in range(start_iteration, start_iteration + args.iterations):
        if iteration == 0 and not history:
            candidate = baseline_config(args)
            hypothesis = "Baseline GEPA configuration."
            proposal_source = "baseline"
        else:
            candidate = None
            hypothesis = ""
            proposal_source = cast(ProposalPolicy, args.proposal_policy)

            if args.proposal_policy == "llm":
                try:
                    proposal = propose_with_llm(
                        provider=provider,
                        research_model=research_model,
                        current_best=current_best_config,
                        history=history,
                        space=space,
                        dev_size=args.dev_size,
                    )
                    candidate = canonicalize_config(
                        GEPAExperimentConfig(
                            max_metric_calls=proposal.config.max_metric_calls,
                            reflection_minibatch_size=proposal.config.reflection_minibatch_size,
                            candidate_selection_strategy=proposal.config.candidate_selection_strategy,
                            seed=proposal.config.seed,
                            n_runners=proposal.config.n_runners,
                        ),
                        space,
                        args.dev_size,
                    )
                    hypothesis = proposal.hypothesis
                except Exception as exc:  # pragma: no cover - defensive fallback for runtime environments
                    logger.warning("LLM proposal failed, falling back to mutation: %s", exc)
                    candidate = mutate_config(current_best_config, rng, space, args.dev_size)
                    hypothesis = f"Fallback mutation after LLM proposal failure: {exc}"
                    proposal_source = "mutation_fallback"
            elif args.proposal_policy == "mutation":
                candidate = mutate_config(current_best_config, rng, space, args.dev_size)
                hypothesis = "Mutate a few GEPA knobs around the current best configuration."
            else:
                candidate = random_config(rng, space, args.dev_size)
                hypothesis = "Randomly sample a fresh GEPA configuration from the search space."

            assert candidate is not None
            candidate = unique_candidate(
                candidate,
                current_best=current_best_config,
                seen=seen,
                rng=rng,
                space=space,
                dev_size=args.dev_size,
            )

        candidate = canonicalize_config(candidate, space, args.dev_size)
        candidate_key = config_to_key(candidate)
        seen.add(candidate_key)

        this_trial_dir = trial_dir(run_dir, iteration)
        ensure_log_dir(this_trial_dir)

        try:
            result = run_gepa_experiment(
                provider=provider,
                experiment_config=candidate,
                train_size=args.train_size,
                dev_size=args.dev_size,
                train_seed=args.train_seed,
                eval_seed=args.eval_seed,
                holdout_size=args.holdout_size,
                holdout_eval_seed=args.holdout_eval_seed,
                holdout_split=cast(Literal["dev", "test"], args.holdout_split),
                artifact_dir=this_trial_dir,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_name=f"{args.wandb_name_prefix}-{iteration:03d}",
            )
            accepted = result.holdout_mean_reward > current_best_score + args.min_improvement
            trial = build_trial_summary(
                iteration=iteration,
                accepted=accepted,
                proposal_source=proposal_source,
                hypothesis=hypothesis,
                config=candidate,
                result=result,
            )
            append_history_entry(run_dir, trial)
            history.append(trial.to_dict())

            print(
                f"[trial {iteration:03d}] holdout={result.holdout_mean_reward:.4f} "
                f"inner={result.inner_val_mean_reward:.4f} accepted={accepted} "
                f"config={asdict(candidate)}"
            )

            if accepted:
                current_best_config = candidate
                current_best_score = result.holdout_mean_reward
                write_best_result(run_dir, trial)
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            trial = build_trial_summary(
                iteration=iteration,
                accepted=False,
                proposal_source=proposal_source,
                hypothesis=hypothesis,
                config=candidate,
                error=str(exc),
            )
            append_history_entry(run_dir, trial)
            history.append(trial.to_dict())
            print(f"[trial {iteration:03d}] failed error={exc} config={asdict(candidate)}")

    final_best = history_best_entry(history)
    print("\nBest result:")
    if final_best is None:
        print("No successful trials were recorded.")
        return

    print(json.dumps(final_best, indent=2))


if __name__ == "__main__":
    main()
