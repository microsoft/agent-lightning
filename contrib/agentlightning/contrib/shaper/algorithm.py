# Copyright (c) Microsoft. All rights reserved.

"""Agent Lightning implementation of two-stage SHAPER artifact evolution."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import Counter
from typing import Any, Callable, Dict, Generic, Iterable, Literal, Optional, Sequence, TypeVar, cast

from openai import AsyncOpenAI

from agentlightning.adapter import TraceAdapter
from agentlightning.algorithm.base import Algorithm
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset, NamedResources, PromptTemplate, Rollout, RolloutMode, TaskInput

from .prompting import load_prompt, parse_json_object
from .roles import OptimizationStage, OptimizerRequestContext, RoleRequest, SHAPERRoleProtocol
from .sandbox import PythonHarnessValidator
from .trace import SHAPERTraceAdapter
from .types import (
    ArtifactCandidate,
    CandidateEvaluation,
    EpisodeSummary,
    EpisodeTrace,
    OptimizationEvent,
    RoundCritique,
    RoundRecord,
)

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task")
SkillValidator = Callable[[str], Sequence[str]]


class IncomparableCandidateError(RuntimeError):
    """A candidate cannot be ranked on the fixed validation denominator."""


class RolloutInfrastructureError(RuntimeError):
    """Rollout orchestration failed independently of an artifact's behavior."""


def validate_nonempty_skill(source: str) -> Sequence[str]:
    """Apply the benchmark-agnostic minimum contract for planner skills."""

    return [] if source.strip() else ["Skill must not be empty."]


def _diagnostic_context(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Separate multimodal context images from a compact JSON description."""

    images: list[dict[str, Any]] = []

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(child) for child in cast(list[Any], item)]
        if not isinstance(item, dict):
            return item
        mapping = cast(dict[str, Any], item)
        if mapping.get("type") == "image_url" and isinstance(mapping.get("image_url"), dict):
            image_value = cast(dict[str, Any], mapping["image_url"])
            raw_url = image_value.get("url")
            if isinstance(raw_url, str):
                images.append(mapping)
                return {
                    "type": "image_url",
                    "image_url": {"url": "<multimodal image supplied below>"},
                }
        return {str(key): visit(child) for key, child in mapping.items()}

    return visit(value), images


DEFAULT_HARNESS_CONTRACT = """The module must define:
    def build_context(history): ...

history is a bounded list of observable round dictionaries. A typical record
contains task_instruction, round_index, planner_response, command,
observation_before, observation_after, action_result, execution_steps, and
runtime_errors. Benchmark integrations may add documented observable fields.
The function must return a deterministic JSON-serializable context payload.
It must not read hidden simulator state, ground-truth answers, or external
state, and its work and output size must remain bounded."""


class _BatchSampler(Generic[T_task]):
    """Deterministic epoch sampler used for rollout-gradient minibatches."""

    def __init__(self, dataset: Sequence[T_task], batch_size: int, seed: int) -> None:
        if not dataset:
            raise ValueError("Training dataset must not be empty.")
        if batch_size < 1:
            raise ValueError("gradient_batch_size must be at least 1.")
        self._dataset = dataset
        self._batch_size = min(batch_size, len(dataset))
        self._random = random.Random(seed)
        self._indices: list[int] = []
        self._cursor = 0

    def next(self) -> list[T_task]:
        """Return the next minibatch, reshuffling only at epoch boundaries."""

        if self._cursor + self._batch_size > len(self._indices):
            self._indices = list(range(len(self._dataset)))
            self._random.shuffle(self._indices)
            self._cursor = 0
        selected = self._indices[self._cursor : self._cursor + self._batch_size]
        self._cursor += self._batch_size
        return [self._dataset[index] for index in selected]


class SHAPER(Algorithm, Generic[T_task]):
    """Evolve a textual skill and context-code harness around frozen agents.

    SHAPER follows a fixed two-stage schedule. The first stage updates only the
    skill while holding the seed harness fixed. The second stage freezes the
    selected skill and updates only the harness. Every proposal is selected by
    reward on one fixed validation set, and incumbent candidates remain eligible
    in each top-K update.

    Agent Lightning currently has a closed resource union. Both artifacts are
    therefore transported as [`PromptTemplate`][agentlightning.PromptTemplate]
    resources; harness consumers read ``resource.template`` as Python source and
    never call ``format`` on it.
    """

    def __init__(
        self,
        async_openai_client: AsyncOpenAI,
        *,
        model: str,
        skill_resource_name: str = "skill",
        harness_resource_name: str = "harness",
        gradient_batch_size: int = 4,
        validation_size: Optional[int] = None,
        beam_width: int = 3,
        branch_factor: int = 2,
        skill_rounds: int = 2,
        harness_rounds: int = 2,
        rollout_batch_timeout: float = 3600.0,
        optimizer_temperature: float = 0.7,
        role_max_completion_tokens: int = 4096,
        role_extra_body: Optional[Dict[str, Any]] = None,
        api_retries: int = 3,
        artifact_repair_attempts: int = 1,
        random_seed: int = 0,
        skill_validator: Optional[SkillValidator] = None,
        harness_validator: Optional[PythonHarnessValidator] = None,
        harness_contract: str = DEFAULT_HARNESS_CONTRACT,
        judger_prompt: Optional[str] = None,
        summarizer_prompt: Optional[str] = None,
        skill_optimizer_prompt: Optional[str] = None,
        harness_optimizer_prompt: Optional[str] = None,
        role_protocol: Optional[SHAPERRoleProtocol] = None,
    ) -> None:
        """Initialize SHAPER.

        Args:
            async_openai_client: Client used by judger, summarizer, and artifact optimizer roles.
            model: One frozen model identifier shared by all evolution roles.
            skill_resource_name: Named-resource key containing the textual skill.
            harness_resource_name: Named-resource key containing harness Python source.
            gradient_batch_size: Rollouts summarized into each textual gradient.
            validation_size: Optional fixed subset size; ``None`` uses all validation tasks.
            beam_width: Number of incumbents retained after each validation round.
            branch_factor: Number of proposals sampled from each beam parent.
            skill_rounds: Number of skill-only beam rounds.
            harness_rounds: Number of harness-only beam rounds.
            rollout_batch_timeout: Wall-clock allowance for one concurrent
                rollout wave. The total batch allowance scales with validation
                size and Trainer runner count.
            optimizer_temperature: Sampling temperature for artifact proposals.
            role_max_completion_tokens: Output-token limit for evolution-role calls.
            role_extra_body: Provider-specific request fields shared by all
                SHAPER role-model calls.
            api_retries: Number of attempts for a failed role-model request.
            artifact_repair_attempts: Extra optimizer calls after skill or harness
                validation failure.
            random_seed: Seed for train minibatches and fixed validation subsampling.
            skill_validator: Benchmark-owned validator for generated planner skills.
            harness_validator: Validator for generated harness source.
            harness_contract: Observable input schema and output contract supplied
                to the harness optimizer.
            judger_prompt: Optional replacement for the bundled round-judger prompt.
            summarizer_prompt: Optional replacement for the bundled episode prompt.
            skill_optimizer_prompt: Optional replacement skill-optimizer prompt.
            harness_optimizer_prompt: Optional replacement harness-optimizer prompt.
            role_protocol: Optional benchmark-specific role formatting and
                parsing protocol. When supplied, it owns all diagnostic and
                optimizer requests while SHAPER retains artifact validation,
                candidate evaluation, and selection.
        """

        if skill_resource_name == harness_resource_name:
            raise ValueError("Skill and harness resource names must differ.")
        for name, value in {
            "gradient_batch_size": gradient_batch_size,
            "beam_width": beam_width,
            "branch_factor": branch_factor,
            "role_max_completion_tokens": role_max_completion_tokens,
            "api_retries": api_retries,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
        if skill_rounds < 0 or harness_rounds < 0 or skill_rounds + harness_rounds < 1:
            raise ValueError("At least one non-negative skill or harness round is required.")
        if validation_size is not None and validation_size < 1:
            raise ValueError("validation_size must be at least 1 when provided.")
        if rollout_batch_timeout <= 0:
            raise ValueError("rollout_batch_timeout must be positive.")
        if artifact_repair_attempts < 0:
            raise ValueError("artifact_repair_attempts must not be negative.")
        if not harness_contract.strip():
            raise ValueError("harness_contract must not be empty.")

        self.async_openai_client = async_openai_client
        self.model = model
        self.skill_resource_name = skill_resource_name
        self.harness_resource_name = harness_resource_name
        self.gradient_batch_size = gradient_batch_size
        self.validation_size = validation_size
        self.beam_width = beam_width
        self.branch_factor = branch_factor
        self.skill_rounds = skill_rounds
        self.harness_rounds = harness_rounds
        self.rollout_batch_timeout = rollout_batch_timeout
        self.optimizer_temperature = optimizer_temperature
        self.role_max_completion_tokens = role_max_completion_tokens
        self.role_extra_body = dict(role_extra_body or {})
        self.api_retries = api_retries
        self.artifact_repair_attempts = artifact_repair_attempts
        self.random_seed = random_seed
        self.skill_validator = skill_validator or validate_nonempty_skill
        self.harness_validator = harness_validator or PythonHarnessValidator()
        self.harness_contract = harness_contract.strip()

        self._version_counter = 0
        self._seed_resources: Optional[NamedResources] = None
        self._seed_harness: Optional[PromptTemplate] = None
        self._best_candidate: Optional[ArtifactCandidate] = None
        self._validation_cache: dict[str, CandidateEvaluation] = {}
        self._optimization_history: list[OptimizationEvent] = []

        self._judger_prompt = judger_prompt or load_prompt("round_judger.txt")
        self._summarizer_prompt = summarizer_prompt or load_prompt("episode_summarizer.txt")
        self._skill_optimizer_prompt = skill_optimizer_prompt or load_prompt("skill_optimizer.txt")
        self._harness_optimizer_prompt = harness_optimizer_prompt or load_prompt("harness_optimizer.txt")
        self._role_protocol = role_protocol

    def get_best_candidate(self) -> ArtifactCandidate:
        """Return the best validation candidate encountered across both stages."""

        if self._best_candidate is None:
            raise ValueError("SHAPER has not completed an initial validation.")
        return self._best_candidate.model_copy(deep=True)

    def get_best_resources(self) -> NamedResources:
        """Return frozen initial resources with the best artifacts installed."""

        best = self.get_best_candidate()
        return self._resources_for_candidate(best)

    def get_optimization_history(self) -> list[OptimizationEvent]:
        """Return a detached copy of proposal, validation, and rejection history."""

        return [event.model_copy(deep=True) for event in self._optimization_history]

    def _get_trace_adapter(self) -> SHAPERTraceAdapter:
        adapter: TraceAdapter[Any] = self.get_adapter()
        if not isinstance(adapter, SHAPERTraceAdapter):
            raise ValueError("SHAPER requires SHAPERTraceAdapter as the Trainer adapter.")
        return adapter

    def _initial_artifacts(self) -> tuple[PromptTemplate, PromptTemplate]:
        resources = self.get_initial_resources()
        if resources is None:
            raise ValueError("SHAPER requires initial_resources with skill and harness PromptTemplates.")
        skill = resources.get(self.skill_resource_name)
        harness = resources.get(self.harness_resource_name)
        if not isinstance(skill, PromptTemplate):
            raise ValueError(f"Resource {self.skill_resource_name!r} must be a PromptTemplate.")
        if not isinstance(harness, PromptTemplate):
            raise ValueError(f"Resource {self.harness_resource_name!r} must be a PromptTemplate.")
        skill_errors = list(self.skill_validator(skill.template))
        if skill_errors:
            raise ValueError("Seed skill failed validation: " + "; ".join(skill_errors))
        harness_validation = self.harness_validator.validate(harness.template)
        if not harness_validation.valid:
            raise ValueError("Seed harness failed validation: " + "; ".join(harness_validation.errors))
        self._seed_resources = dict(resources)
        self._seed_harness = harness
        return skill, harness

    def _new_candidate(
        self,
        *,
        skill: PromptTemplate,
        harness: PromptTemplate,
        stage: Literal["seed", "skill", "harness"],
        parent_version: Optional[str] = None,
        rationale: str = "",
    ) -> ArtifactCandidate:
        version = f"shaper-v{self._version_counter:04d}"
        self._version_counter += 1
        return ArtifactCandidate(
            version=version,
            skill=skill,
            harness=harness,
            stage=stage,
            parent_version=parent_version,
            rationale=rationale,
        )

    def _resources_for_candidate(self, candidate: ArtifactCandidate) -> NamedResources:
        if self._seed_resources is None:
            raise ValueError("Initial resources have not been loaded.")
        resources = dict(self._seed_resources)
        resources[self.skill_resource_name] = candidate.skill
        resources[self.harness_resource_name] = candidate.harness
        return resources

    @staticmethod
    def _materialize_dataset(dataset: Optional[Dataset[T_task]], name: str) -> list[T_task]:
        if dataset is None:
            raise ValueError(f"{name} dataset is required for SHAPER.")
        materialized = [dataset[index] for index in range(len(dataset))]
        if not materialized:
            raise ValueError(f"{name} dataset must not be empty.")
        return materialized

    def _select_fixed_validation(self, dataset: Sequence[T_task]) -> list[T_task]:
        if self.validation_size is None or self.validation_size >= len(dataset):
            return list(dataset)
        rng = random.Random(self.random_seed)
        indices = sorted(rng.sample(range(len(dataset)), self.validation_size))
        return [dataset[index] for index in indices]

    def _rollout_batch_allowance(self, rollout_count: int) -> float:
        """Scale a per-wave allowance to the number of configured runners."""

        if rollout_count < 1:
            raise ValueError("rollout_count must be at least 1.")
        try:
            n_runners = self.get_trainer().n_runners
        except ValueError:
            # Direct unit integrations may attach a store without a Trainer.
            n_runners = 1
        n_runners = max(1, n_runners)
        waves = (rollout_count + n_runners - 1) // n_runners
        return self.rollout_batch_timeout * waves

    async def _evaluate_candidate(
        self,
        candidate: ArtifactCandidate,
        dataset: Sequence[T_task],
        mode: Literal["train", "val"],
    ) -> CandidateEvaluation:
        if mode == "val" and candidate.version in self._validation_cache:
            return self._validation_cache[candidate.version]

        store = self.get_store()
        resources = self._resources_for_candidate(candidate)
        update = await store.update_resources(candidate.version, resources)
        queued: list[Rollout] = []
        for task in dataset:
            rollout = await store.enqueue_rollout(
                input=cast(TaskInput, task),
                mode=cast(RolloutMode, mode),
                resources_id=update.resources_id,
            )
            queued.append(rollout)

        rollout_ids = [rollout.rollout_id for rollout in queued]
        batch_allowance = self._rollout_batch_allowance(len(rollout_ids))
        deadline = time.monotonic() + batch_allowance
        finished: list[Rollout] = []
        while True:
            finished = list(await store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=0.0))
            if len(finished) >= len(rollout_ids):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(1.0, max(0.01, remaining)))

        finished_by_id = {rollout.rollout_id: rollout for rollout in finished}
        unfinished_ids = [rollout_id for rollout_id in rollout_ids if rollout_id not in finished_by_id]
        if unfinished_ids:
            sample = ", ".join(unfinished_ids[:3])
            suffix = "" if len(unfinished_ids) <= 3 else ", ..."
            raise RolloutInfrastructureError(
                f"Candidate {candidate.version} exceeded its {batch_allowance:.1f}s rollout "
                f"batch allowance with {len(unfinished_ids)}/{len(rollout_ids)} unfinished "
                f"rollout(s): {sample}{suffix}. Training is aborted so unfinished simulator "
                "work cannot contaminate a later candidate."
            )

        failed_ids = [rollout.rollout_id for rollout in finished if rollout.status != "succeeded"]
        if failed_ids:
            sample = ", ".join(failed_ids[:3])
            suffix = "" if len(failed_ids) <= 3 else ", ..."
            raise RolloutInfrastructureError(
                f"Candidate {candidate.version} had {len(failed_ids)} non-succeeded runner "
                f"rollout(s): {sample}{suffix}. Benchmark-owned valid failures must return "
                "zero reward and observable episode metadata rather than crash the runner."
            )

        adapter = self._get_trace_adapter()
        traces: list[EpisodeTrace] = []

        for queued_rollout in queued:
            completed = finished_by_id.get(queued_rollout.rollout_id)
            if completed is None:
                raise AssertionError("All rollouts were checked for completion above.")

            spans = await store.query_spans(completed.rollout_id)
            trace = adapter.adapt(spans).model_copy(
                update={
                    "rollout_id": completed.rollout_id,
                    "task": completed.input,
                    "status": completed.status,
                }
            )
            traces.append(trace)

        valid_traces = [trace for trace in traces if not trace.metadata.environment_invalid]
        if not valid_traces:
            raise IncomparableCandidateError(
                f"Candidate {candidate.version} produced no valid {mode} rollouts; "
                "simulator-invalid runs cannot define an artifact score."
            )
        if mode == "val" and len(valid_traces) != len(traces):
            invalid_count = len(traces) - len(valid_traces)
            raise IncomparableCandidateError(
                f"Candidate {candidate.version} produced {invalid_count} simulator-invalid "
                f"validation rollout(s). Every candidate must be scored on the same fixed "
                "validation tasks; this candidate is not comparable."
            )
        reward_sum = sum(float(trace.final_reward or 0.0) for trace in valid_traces)
        evaluation = CandidateEvaluation(
            candidate_version=candidate.version,
            mode=mode,
            requested_rollouts=len(dataset),
            finished_rollouts=len(finished_by_id),
            valid_rollouts=len(valid_traces),
            score=reward_sum / len(valid_traces),
            traces=traces,
        )
        if mode == "val":
            candidate.validation_score = evaluation.score
            self._validation_cache[candidate.version] = evaluation
            self._record_candidate_score(candidate)
            self._consider_historical_best(candidate)

        logger.info(
            "[%s] %s score %.4f (%d valid, %d/%d rollouts finished)",
            candidate.version,
            mode,
            evaluation.score,
            evaluation.valid_rollouts,
            evaluation.finished_rollouts,
            evaluation.requested_rollouts,
        )
        return evaluation

    def _record_candidate_score(self, candidate: ArtifactCandidate) -> None:
        for index in range(len(self._optimization_history) - 1, -1, -1):
            event = self._optimization_history[index]
            if event.candidate_version == candidate.version:
                self._optimization_history[index] = event.model_copy(
                    update={"validation_score": candidate.validation_score}
                )
                break

    def _record_candidate_validation_error(self, candidate: ArtifactCandidate, error: str) -> None:
        for index in range(len(self._optimization_history) - 1, -1, -1):
            event = self._optimization_history[index]
            if event.candidate_version == candidate.version:
                self._optimization_history[index] = event.model_copy(update={"validation_error": error})
                break

    def _consider_historical_best(self, candidate: ArtifactCandidate) -> None:
        score = candidate.validation_score
        if score is None:
            return
        if self._best_candidate is None or score > cast(float, self._best_candidate.validation_score):
            self._best_candidate = candidate.model_copy(deep=True)
            logger.info("Historical best is now %s at %.4f", candidate.version, score)

    async def _complete_role(self, role_request: RoleRequest) -> str:
        """Execute one role request with shared provider and retry settings."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": role_request.system_prompt},
            {"role": "user", "content": role_request.user_content},
        ]
        last_error: Optional[BaseException] = None
        for attempt in range(self.api_retries):
            try:
                request: Dict[str, Any] = {
                    "model": self.model,
                    "messages": cast(Any, messages),
                    "temperature": role_request.temperature,
                    "max_completion_tokens": self.role_max_completion_tokens,
                }
                if role_request.response_format == "json_object":
                    request["response_format"] = {"type": "json_object"}
                if self.role_extra_body:
                    request["extra_body"] = self.role_extra_body
                response = cast(
                    Any,
                    await self.async_openai_client.chat.completions.create(
                        **request,
                    ),
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Role model returned empty content.")
                return content.strip()
            except (Exception, asyncio.CancelledError) as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                last_error = exc
                if attempt + 1 < self.api_retries:
                    await asyncio.sleep(min(2**attempt, 4))
        raise RuntimeError(f"SHAPER role model failed after {self.api_retries} attempts: {last_error}")

    async def _complete_json(
        self,
        *,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        content = await self._complete_role(
            RoleRequest(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=temperature,
                response_format="json_object",
            )
        )
        return parse_json_object(content)

    async def _judge_round(self, record: RoundRecord) -> RoundCritique:
        context_description, context_images = _diagnostic_context(record.context_payload)
        header = {
            "round_index": record.round_index,
            "task_instruction": record.task_instruction,
            "planner_response": record.planner_response,
            "command": record.command,
            "context_payload": context_description,
            "execution_steps": record.execution_steps,
            "action_result": record.action_result,
            "runtime_errors": record.runtime_errors,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "ROUND RECORD\n" + json.dumps(header, ensure_ascii=False, default=str),
            },
            {"type": "text", "text": "IMAGES ROUTED BY THE CONTEXT HARNESS"},
            *context_images,
            {"type": "text", "text": "OBSERVATION BEFORE EXECUTION"},
            *record.observation_before,
            {"type": "text", "text": "OBSERVATION AFTER EXECUTION"},
            *record.observation_after,
        ]
        try:
            payload = await self._complete_json(
                system_prompt=self._judger_prompt,
                user_content=content,
                temperature=0.0,
            )
            payload["round_index"] = record.round_index
            return RoundCritique.model_validate(payload)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Round judger failed for round %d: %s", record.round_index, exc)
            return RoundCritique(
                round_index=record.round_index,
                progress="unclear",
                progress_score=0.0,
                observable_change="Judger output unavailable.",
                command_assessment="Unknown.",
                reasoning_assessment="Unknown.",
                context_assessment="Unknown.",
                likely_cause=f"Diagnostic failure: {exc}",
                suggested_fix="Do not infer an artifact change from this round alone.",
            )

    async def _summarize_episode(
        self,
        trace: EpisodeTrace,
        critiques: Sequence[RoundCritique],
    ) -> EpisodeSummary:
        total_steps = sum(record.execution_steps for record in trace.rounds)
        terminal_context = _diagnostic_context(trace.rounds[-1].context_payload)[0] if trace.rounds else None
        task_instruction = trace.rounds[0].task_instruction if trace.rounds else ""
        user_payload = {
            "rollout_id": trace.rollout_id,
            "task_instruction": task_instruction,
            "status": trace.status,
            "environment_reward": float(trace.final_reward or 0.0),
            "environment_invalid": trace.metadata.environment_invalid,
            "termination_reason": trace.metadata.termination_reason,
            "commands": [record.command for record in trace.rounds],
            "execution_steps": total_steps,
            "runtime_errors": [
                *trace.metadata.runtime_errors,
                *(error for record in trace.rounds for error in record.runtime_errors),
                *trace.adapter_errors,
            ],
            "terminal_context_payload": terminal_context,
            "round_critiques": [critique.model_dump(mode="json") for critique in critiques],
        }
        try:
            payload = await self._complete_json(
                system_prompt=self._summarizer_prompt,
                user_content=json.dumps(user_payload, ensure_ascii=False, default=str),
                temperature=0.0,
            )
            payload["rollout_id"] = trace.rollout_id
            payload["reward"] = float(trace.final_reward or 0.0)
            payload["environment_invalid"] = trace.metadata.environment_invalid
            return EpisodeSummary.model_validate(payload)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Episode summarizer failed for %s: %s", trace.rollout_id, exc)
            return EpisodeSummary(
                rollout_id=trace.rollout_id,
                reward=float(trace.final_reward or 0.0),
                environment_invalid=trace.metadata.environment_invalid,
                instruction_fidelity="Unavailable.",
                progress_and_outcome=f"Reward={float(trace.final_reward or 0.0):.3f}.",
                repetition_or_recovery="Unavailable.",
                decomposition_quality="Unavailable.",
                context_effectiveness="Unavailable.",
                root_cause=f"Diagnostic failure: {exc}",
                actionable_change="Do not infer an artifact change from this episode alone.",
            )

    async def _diagnose_episode(self, trace: EpisodeTrace) -> EpisodeSummary:
        critiques = await asyncio.gather(*(self._judge_round(record) for record in trace.rounds))
        return await self._summarize_episode(trace, critiques)

    async def _build_textual_gradient(self, evaluation: CandidateEvaluation) -> Any:
        if self._role_protocol is not None:
            return await self._role_protocol.build_textual_gradient(
                evaluation,
                self._complete_role,
            )

        valid_traces = [trace for trace in evaluation.traces if not trace.metadata.environment_invalid]
        summaries = await asyncio.gather(*(self._diagnose_episode(trace) for trace in valid_traces))
        rewards = [float(trace.final_reward or 0.0) for trace in valid_traces]
        commands = [record.command for trace in valid_traces for record in trace.rounds]
        runtime_errors = [
            error
            for trace in valid_traces
            for error in [
                *trace.metadata.runtime_errors,
                *trace.adapter_errors,
                *(item for record in trace.rounds for item in record.runtime_errors),
            ]
        ]
        statistics = {
            "requested_rollouts": evaluation.requested_rollouts,
            "finished_rollouts": evaluation.finished_rollouts,
            "valid_rollouts": evaluation.valid_rollouts,
            "mean_reward": evaluation.score,
            "successful_rollouts": sum(reward >= 1.0 for reward in rewards),
            "environment_invalid_rollouts": sum(trace.metadata.environment_invalid for trace in evaluation.traces),
            "command_frequencies": dict(Counter(commands).most_common(20)),
            "runtime_error_frequencies": dict(Counter(runtime_errors).most_common(20)),
        }
        return json.dumps(
            {
                "episode_summaries": [summary.model_dump(mode="json") for summary in summaries],
                "aggregate_statistics": statistics,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _history_text(self) -> str:
        events = [event.model_dump(mode="json") for event in self._optimization_history[-30:]]
        return json.dumps(events, ensure_ascii=False, indent=2)

    async def _propose_artifact(
        self,
        *,
        parent: ArtifactCandidate,
        stage: OptimizationStage,
        textual_gradient: Any,
        round_index: int,
        branch_index: int,
    ) -> Optional[ArtifactCandidate]:
        common = ""
        system_prompt = ""
        if self._role_protocol is None:
            frozen_note = (
                "The seed harness below is fixed in the skill stage."
                if stage == "skill"
                else "The selected skill below is fixed in the harness stage."
            )
            common = (
                f"ROUND: {round_index}\nBRANCH: {branch_index}\nSTAGE: {stage}\n\n"
                f"{frozen_note}\n\nCURRENT SKILL\n=============\n{parent.skill.template}\n\n"
                f"CURRENT HARNESS\n===============\n{parent.harness.template}\n\n"
                f"ROLLOUT-DERIVED TEXTUAL GRADIENT\n================================\n{textual_gradient}\n\n"
                f"OPTIMIZATION HISTORY\n====================\n{self._history_text()}"
            )
            if stage == "harness":
                common += (
                    "\n\nHARNESS VALIDATION CONTRACT\n===========================\n" + self.harness_contract + "\n\n"
                    f"Define one synchronous {self.harness_validator.function_name} function. "
                    f"The default smoke arguments are {list(self.harness_validator.smoke_args)!r}. "
                    "Unsupported imports, dynamic execution, reflection, and arbitrary file I/O are rejected; "
                    "runtime CPU, memory, output, and wall-clock limits contain expensive work."
                )
            system_prompt = self._skill_optimizer_prompt if stage == "skill" else self._harness_optimizer_prompt

        feedback = ""
        for repair_index in range(self.artifact_repair_attempts + 1):
            try:
                if self._role_protocol is None:
                    payload = await self._complete_json(
                        system_prompt=system_prompt,
                        user_content=common + feedback,
                        temperature=self.optimizer_temperature,
                    )
                    rationale = payload.get("rationale")
                    artifact = payload.get("new_artifact")
                else:
                    request = self._role_protocol.build_optimizer_request(
                        OptimizerRequestContext(
                            parent=parent,
                            stage=stage,
                            textual_gradient=textual_gradient,
                            round_index=round_index,
                            branch_index=branch_index,
                            optimization_history=tuple(self._optimization_history),
                            harness_contract=self.harness_contract,
                            harness_function_name=self.harness_validator.function_name,
                            harness_smoke_args=tuple(self.harness_validator.smoke_args),
                            validation_feedback=feedback,
                        )
                    )
                    raw_response = await self._complete_role(request)
                    proposal = self._role_protocol.parse_optimizer_response(stage, raw_response)
                    rationale = proposal.rationale
                    artifact = proposal.artifact
            except (RuntimeError, ValueError) as exc:
                if isinstance(exc, ValueError) and repair_index < self.artifact_repair_attempts:
                    feedback = (
                        "\n\nOUTPUT PARSE ERROR\n"
                        + str(exc)
                        + "\nReturn one complete replacement artifact in the required output format."
                    )
                    continue
                self._optimization_history.append(
                    OptimizationEvent(
                        round_index=round_index,
                        stage=stage,
                        parent_version=parent.version,
                        rationale="Proposal generation failed.",
                        validation_error=str(exc),
                    )
                )
                logger.warning("Artifact proposal failed for %s: %s", parent.version, exc)
                return None
            if not isinstance(rationale, str) or not isinstance(artifact, str) or not artifact.strip():
                feedback = (
                    "\n\nVALIDATION ERROR\n"
                    "Return non-empty string fields rationale and new_artifact. "
                    "new_artifact must be the complete replacement artifact, not a diff or wrapper."
                )
                continue

            validation_errors: list[str] = []
            if stage == "skill":
                validation_errors = list(self.skill_validator(artifact))
            else:
                validation = self.harness_validator.validate(artifact)
                if not validation.valid:
                    validation_errors = list(validation.errors)
            if validation_errors:
                error_text = "; ".join(validation_errors)
                self._optimization_history.append(
                    OptimizationEvent(
                        round_index=round_index,
                        stage=stage,
                        parent_version=parent.version,
                        rationale=rationale,
                        validation_error=error_text,
                    )
                )
                feedback = (
                    f"\n\n{stage.upper()} VALIDATION FAILED\n"
                    + error_text
                    + "\nRepair the invalid artifact below. Make the smallest change needed to satisfy "
                    "the validator; do not redesign it, add wrappers, or switch artifact types.\n\n"
                    "PREVIOUS INVALID ARTIFACT\n=========================\n"
                    + artifact
                    + "\n\nEND PREVIOUS INVALID ARTIFACT\n"
                    "Return the corrected complete artifact while preserving evidence-backed intent."
                )
                if repair_index < self.artifact_repair_attempts:
                    continue
                return None

            if stage == "skill":
                skill = PromptTemplate(template=artifact.strip(), engine="f-string")
                harness = parent.harness
            else:
                skill = parent.skill
                harness = PromptTemplate(template=artifact.strip(), engine="f-string")
            candidate = self._new_candidate(
                skill=skill,
                harness=harness,
                stage=stage,
                parent_version=parent.version,
                rationale=rationale,
            )
            self._optimization_history.append(
                OptimizationEvent(
                    round_index=round_index,
                    stage=stage,
                    parent_version=parent.version,
                    candidate_version=candidate.version,
                    rationale=rationale,
                )
            )
            return candidate
        self._optimization_history.append(
            OptimizationEvent(
                round_index=round_index,
                stage=stage,
                parent_version=parent.version,
                rationale="Malformed proposal rejected.",
                validation_error=feedback.strip() or "Optimizer returned an invalid artifact payload.",
            )
        )
        return None

    async def _generate_children(
        self,
        *,
        beam: Sequence[ArtifactCandidate],
        stage: OptimizationStage,
        round_index: int,
        train_sampler: _BatchSampler[T_task],
    ) -> list[ArtifactCandidate]:
        children: list[ArtifactCandidate] = []
        for parent in beam:
            evaluation = await self._evaluate_candidate(parent, train_sampler.next(), "train")
            textual_gradient = await self._build_textual_gradient(evaluation)
            proposed = await asyncio.gather(
                *(
                    self._propose_artifact(
                        parent=parent,
                        stage=stage,
                        textual_gradient=textual_gradient,
                        round_index=round_index,
                        branch_index=branch_index,
                    )
                    for branch_index in range(self.branch_factor)
                )
            )
            children.extend(candidate for candidate in proposed if candidate is not None)

        seen = {candidate.artifact_key() for candidate in beam}
        unique: list[ArtifactCandidate] = []
        for candidate in children:
            key = candidate.artifact_key()
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    async def _select_beam(
        self,
        candidates: Iterable[ArtifactCandidate],
        fixed_validation: Sequence[T_task],
    ) -> list[ArtifactCandidate]:
        candidate_list = list(candidates)
        comparable: list[ArtifactCandidate] = []
        for candidate in candidate_list:
            if candidate.validation_score is None:
                try:
                    await self._evaluate_candidate(candidate, fixed_validation, "val")
                except IncomparableCandidateError as exc:
                    self._record_candidate_validation_error(candidate, str(exc))
                    logger.warning("Rejecting incomparable candidate %s: %s", candidate.version, exc)
                    continue
            comparable.append(candidate)
        comparable.sort(
            key=lambda candidate: cast(float, candidate.validation_score),
            reverse=True,
        )
        if not comparable:
            raise RuntimeError("SHAPER beam became empty because no candidate had a comparable validation score.")
        return comparable[: self.beam_width]

    async def run(
        self,
        train_dataset: Optional[Dataset[T_task]] = None,
        val_dataset: Optional[Dataset[T_task]] = None,
    ) -> None:
        """Run hierarchical diagnosis and two-stage top-K artifact evolution."""

        skill, harness = self._initial_artifacts()
        self._get_trace_adapter()
        training = self._materialize_dataset(train_dataset, "Training")
        validation = self._materialize_dataset(val_dataset, "Validation")
        fixed_validation = self._select_fixed_validation(validation)
        train_sampler = _BatchSampler(training, self.gradient_batch_size, self.random_seed)

        seed = self._new_candidate(skill=skill, harness=harness, stage="seed")
        await self._evaluate_candidate(seed, fixed_validation, "val")
        beam: list[ArtifactCandidate] = [seed]

        round_index = 0
        for _ in range(self.skill_rounds):
            children = await self._generate_children(
                beam=beam,
                stage="skill",
                round_index=round_index,
                train_sampler=train_sampler,
            )
            beam = await self._select_beam([*beam, *children], fixed_validation)
            round_index += 1

        selected_skill = beam[0]
        if self._seed_harness is None:
            raise RuntimeError("Seed harness was not initialized.")
        if selected_skill.harness.template != self._seed_harness.template:
            raise RuntimeError("Skill stage modified the frozen seed harness.")

        harness_seed = selected_skill
        beam = [harness_seed]
        frozen_skill_text = harness_seed.skill.template
        for _ in range(self.harness_rounds):
            children = await self._generate_children(
                beam=beam,
                stage="harness",
                round_index=round_index,
                train_sampler=train_sampler,
            )
            if any(candidate.skill.template != frozen_skill_text for candidate in children):
                raise RuntimeError("Harness stage modified the frozen selected skill.")
            beam = await self._select_beam([*beam, *children], fixed_validation)
            round_index += 1

        best = self.get_best_candidate()
        store: LightningStore = self.get_store()
        await store.update_resources(best.version, self._resources_for_candidate(best))
        logger.info(
            "SHAPER complete: best=%s score=%.4f stage=%s",
            best.version,
            cast(float, best.validation_score),
            best.stage,
        )
