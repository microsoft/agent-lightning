# Copyright (c) Microsoft. All rights reserved.

"""Paper-faithful SHAPER role protocol for ESI-Bench."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from agentlightning.contrib.shaper import (
    ArtifactProposal,
    CandidateEvaluation,
    EpisodeTrace,
    OptimizationStage,
    OptimizerRequestContext,
    RoleCompleter,
    RoleRequest,
    RoundRecord,
    parse_json_object,
)

from ..common import load_text

_FAILURE_LABELS = {
    "none",
    "missing_evidence",
    "lost_cross_view_evidence",
    "reference_ignored",
    "action_blindness",
    "invalid_action",
    "viewpoint_deadlock",
    "premature_commitment",
    "reasoning_error",
    "confidence_miscalibration",
    "budget_exhaustion",
    "task_contract_violation",
    "environment_failure",
    "other",
}

_AGL_HARNESS_ADAPTER = """
## Agent Lightning execution adapter (authoritative for this contrib)

The paper prompt above describes the original in-process ESI-Bench context
module. This contrib executes optimizer-generated code in an isolated JSON-only
worker. Preserve the same evidence-selection objective, but obey this concrete
interface:

- Define exactly `def build_context(records)`.
- `records` contains only the observable, JSON-serializable fields documented
  in the validation contract supplied in the user message.
- Official and derived images are already OpenAI `image_url` parts nested in
  each observation. Reuse those parts directly; do not read or write paths.
- No `Path`, `numpy`, `cv2`, network client, process state, or simulator object
  is available. Imports and external file access are rejected.
- Return a string or a bounded list of OpenAI `text` / `image_url` parts.
- The validation contract supplied in the user message overrides incompatible
  low-level runtime details in the original prompt.
""".strip()


@dataclass(frozen=True)
class _JudgedTrace:
    trace: EpisodeTrace
    judgement: dict[str, Any]


@dataclass(frozen=True)
class _ESIBenchGradient:
    evaluation: CandidateEvaluation
    judged: tuple[_JudgedTrace, ...]
    batch_summary: str
    n_correct: int
    n_scored: int
    n_invalid: int
    category_summary: str
    execution_statistics: str


def _split_images(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(child) for child in cast(list[Any], item)]
        if not isinstance(item, dict):
            return item
        mapping = cast(dict[str, Any], item)
        if mapping.get("type") == "image_url" and isinstance(mapping.get("image_url"), dict):
            image = cast(dict[str, Any], mapping["image_url"])
            if isinstance(image.get("url"), str):
                images.append(mapping)
                return {"type": "image_url", "image_url": {"url": "<image supplied separately>"}}
        return {str(key): visit(child) for key, child in mapping.items()}

    return visit(value), images


def _image_parts(parts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], part)
        for part in parts
        if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict)
    ]


def _metadata(trace: EpisodeTrace) -> dict[str, Any]:
    return trace.metadata.extra


def _official_correct(trace: EpisodeTrace) -> bool:
    value = _metadata(trace).get("official_correct")
    return bool(value) if isinstance(value, bool) else float(trace.final_reward or 0.0) >= 1.0


def _family(trace: EpisodeTrace) -> str:
    value = str(_metadata(trace).get("task_family", "")).strip()
    return value or "Unknown"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _selected_round_indices(rounds: Sequence[RoundRecord], limit: int = 6) -> list[int]:
    if len(rounds) <= limit:
        return list(range(len(rounds)))
    candidates = [0, 1, len(rounds) // 2, len(rounds) - 2, len(rounds) - 1]
    for index in range(1, len(rounds)):
        if rounds[index].command != rounds[index - 1].command:
            candidates.append(index)
    output: list[int] = []
    for index in candidates:
        if 0 <= index < len(rounds) and index not in output:
            output.append(index)
        if len(output) >= limit:
            break
    return sorted(output)


def _trajectory(trace: EpisodeTrace) -> str:
    lines: list[str] = []
    for record in trace.rounds:
        result = record.action_result
        lines.append(
            "Step "
            + str(record.round_index + 1)
            + "\nReason/Action output: "
            + record.planner_response
            + "\nParsed action: "
            + record.command
            + "\nAction valid: "
            + str(result.get("action_valid", True))
            + "\nAction result: "
            + _json_text(result)
        )
    return "\n\n".join(lines) if lines else "(No executable planner step was recorded.)"


def _reference_images(trace: EpisodeTrace) -> list[dict[str, Any]]:
    raw = _metadata(trace).get("reference_images")
    if not isinstance(raw, list):
        return []
    output: list[dict[str, Any]] = []
    for item in cast(list[Any], raw):
        if not isinstance(item, dict):
            continue
        mapping = cast(dict[str, Any], item)
        image = mapping.get("image")
        if isinstance(image, dict):
            image_mapping = cast(dict[str, Any], image)
            if image_mapping.get("type") == "image_url":
                output.append(image_mapping)
    return output


def _deduplicate_images(images: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for image in images:
        raw = image.get("image_url")
        url = str(cast(dict[str, Any], raw).get("url", "")) if isinstance(raw, dict) else ""
        if url and url not in seen:
            seen.add(url)
            output.append(image)
    return output


class ESIBenchRoleProtocol:
    """Use whole-trajectory diagnosis and development-batch summarization."""

    def __init__(self, prompt_dir: Path) -> None:
        self.judger_system = load_text(prompt_dir, "round_judger.txt")
        self.judger_user = load_text(prompt_dir, "round_judger_user.txt")
        self.summarizer_system = load_text(prompt_dir, "episode_summarizer.txt")
        self.summarizer_user = load_text(prompt_dir, "episode_summarizer_user.txt")
        self.skill_optimizer_system = load_text(prompt_dir, "skill_optimizer.txt")
        self.skill_optimizer_user = load_text(prompt_dir, "skill_optimizer_user.txt")
        self.harness_optimizer_system = load_text(prompt_dir, "harness_optimizer.txt")
        self.harness_optimizer_user = load_text(prompt_dir, "harness_optimizer_user.txt")

    @staticmethod
    def _judger_images(trace: EpisodeTrace) -> list[dict[str, Any]]:
        historical: list[dict[str, Any]] = []
        for index in _selected_round_indices(trace.rounds):
            historical.extend(_image_parts(trace.rounds[index].observation_before))
        if trace.rounds:
            _, context_images = _split_images(trace.rounds[-1].context_payload)
            historical.extend(context_images)
            final = _image_parts(trace.rounds[-1].observation_after)
        else:
            final = []
        content: list[dict[str, Any]] = []
        references = _reference_images(trace)
        if references:
            content.append({"type": "text", "text": "OFFICIAL QUESTION-REFERENCE IMAGES"})
            content.extend(references)
        if historical:
            content.append({"type": "text", "text": "TRAJECTORY AUDIT AND FINAL-CONTEXT IMAGES"})
            content.extend(_deduplicate_images(historical))
        if final:
            content.append({"type": "text", "text": "FINAL/CURRENT OBSERVATIONS"})
            content.extend(_deduplicate_images(final))
        return content

    async def _judge_trace(self, trace: EpisodeTrace, complete: RoleCompleter) -> _JudgedTrace:
        extra = _metadata(trace)
        final_answer_raw = extra.get("final_answer")
        final_answer = cast(dict[str, Any], final_answer_raw) if isinstance(final_answer_raw, dict) else {}
        final_context = trace.rounds[-1].context_payload if trace.rounds else None
        final_context_text, _ = _split_images(final_context)
        environment_diagnostics = {
            "environment_invalid": trace.metadata.environment_invalid,
            "termination_reason": trace.metadata.termination_reason,
            "runtime_errors": [*trace.metadata.runtime_errors, *trace.adapter_errors],
        }
        user_text = self.judger_user.format(
            task_family=_family(trace),
            scene=str(extra.get("scene", "")),
            question_id=str(extra.get("question_id", "")),
            question=str(extra.get("question", "")),
            options=_json_text(extra.get("options")),
            task_contract=str(extra.get("task_contract", "")),
            final_answer=str(final_answer.get("answer", "not sure")),
            final_confidence=final_answer.get("confidence", 0.0),
            termination=trace.metadata.termination_reason,
            ground_truth=_json_text(extra.get("ground_truth")),
            official_correct=str(_official_correct(trace)).lower(),
            environment_diagnostics=_json_text(environment_diagnostics),
            planner_prompt=str(extra.get("planner_skill", "")),
            final_context=_json_text(final_context_text),
            trajectory=_trajectory(trace),
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(self._judger_images(trace))
        raw = await complete(
            RoleRequest(
                system_prompt=self.judger_system,
                user_content=content,
                temperature=1.0,
                response_format="json_object",
            )
        )
        payload = parse_json_object(raw)
        authoritative = _official_correct(trace)
        run_valid = not trace.metadata.environment_invalid
        payload["official_correct"] = authoritative
        payload["run_valid"] = run_valid
        score = payload.get("evidence_progress_score")
        if not run_valid:
            payload["evidence_progress_score"] = None
        elif authoritative:
            payload["evidence_progress_score"] = 1.0
        elif not isinstance(score, (int, float)) or float(score) not in {0.0, 0.25, 0.5, 0.75}:
            raise ValueError(f"ESI-Bench Judger returned invalid evidence_progress_score: {score!r}")
        else:
            payload["evidence_progress_score"] = float(score)
        for key in (
            "evidence_analysis",
            "exploration_analysis",
            "answer_analysis",
            "context_analysis",
            "improvement_signal",
        ):
            if not isinstance(payload.get(key), str):
                raise ValueError(f"ESI-Bench Judger omitted string field {key!r}.")
        failure = str(payload.get("primary_failure", "other"))
        payload["primary_failure"] = failure if failure in _FAILURE_LABELS else "other"
        target = str(payload.get("optimizer_target", "none"))
        payload["optimizer_target"] = target if target in {"prompt", "context", "environment", "none"} else "none"
        if not run_valid:
            payload["primary_failure"] = "environment_failure"
            payload["optimizer_target"] = "environment"
        return _JudgedTrace(trace=trace, judgement=payload)

    @staticmethod
    def _category_summary(evaluation: CandidateEvaluation) -> str:
        totals: dict[str, int] = defaultdict(int)
        correct: dict[str, int] = defaultdict(int)
        invalid: dict[str, int] = defaultdict(int)
        for trace in evaluation.traces:
            family = _family(trace)
            if trace.metadata.environment_invalid:
                invalid[family] += 1
            else:
                totals[family] += 1
                correct[family] += int(_official_correct(trace))
        families = sorted(set(totals) | set(invalid))
        return "\n".join(
            f"{family}: {correct[family]}/{totals[family]} correct; {invalid[family]} environment-invalid"
            for family in families
        )

    @staticmethod
    def _execution_statistics(evaluation: CandidateEvaluation, judged: Sequence[_JudgedTrace]) -> str:
        traces = [item.trace for item in judged]
        steps = [len(trace.rounds) for trace in traces]
        actions = Counter(record.command for trace in traces for record in trace.rounds)
        terminations = Counter(trace.metadata.termination_reason for trace in evaluation.traces)
        invalid_actions = sum(
            not bool(record.action_result.get("action_valid", True)) for trace in traces for record in trace.rounds
        )
        failures = Counter(str(item.judgement.get("primary_failure", "other")) for item in judged)
        payload = {
            "mean_steps": (sum(steps) / len(steps)) if steps else 0.0,
            "min_steps": min(steps, default=0),
            "max_steps": max(steps, default=0),
            "invalid_model_actions": invalid_actions,
            "termination_frequencies": dict(terminations),
            "action_frequencies": dict(actions.most_common(20)),
            "judged_failure_frequencies": dict(failures),
        }
        return _json_text(payload)

    async def build_textual_gradient(
        self,
        evaluation: CandidateEvaluation,
        complete: RoleCompleter,
    ) -> _ESIBenchGradient:
        raw = await asyncio.gather(
            *(self._judge_trace(trace, complete) for trace in evaluation.traces),
            return_exceptions=True,
        )
        judged: list[_JudgedTrace] = []
        for trace, value in zip(evaluation.traces, raw):
            if isinstance(value, BaseException):
                run_valid = not trace.metadata.environment_invalid
                judged.append(
                    _JudgedTrace(
                        trace=trace,
                        judgement={
                            "official_correct": _official_correct(trace),
                            "run_valid": run_valid,
                            "evidence_progress_score": (
                                (1.0 if _official_correct(trace) else 0.0) if run_valid else None
                            ),
                            "evidence_analysis": "Judger output unavailable.",
                            "exploration_analysis": "Judger output unavailable.",
                            "answer_analysis": "Judger output unavailable.",
                            "context_analysis": "Judger output unavailable.",
                            "primary_failure": "other" if run_valid else "environment_failure",
                            "optimizer_target": "none" if run_valid else "environment",
                            "improvement_signal": f"Diagnostic failure: {value}",
                        },
                    )
                )
            else:
                judged.append(value)

        valid = [trace for trace in evaluation.traces if not trace.metadata.environment_invalid]
        valid_judged = [item for item in judged if not item.trace.metadata.environment_invalid]
        n_correct = sum(_official_correct(trace) for trace in valid)
        n_scored = len(valid)
        n_invalid = len(evaluation.traces) - n_scored
        category_summary = self._category_summary(evaluation)
        execution_statistics = self._execution_statistics(evaluation, valid_judged)
        judgements_text = "\n\n".join(_json_text(item.judgement) for item in judged)
        summary_user = self.summarizer_user.format(
            candidate_name=evaluation.candidate_version,
            n_correct=n_correct,
            n_scored=n_scored,
            score=f"{n_correct / max(1, n_scored):.6f}",
            n_valid=n_scored,
            n_environment_invalid=n_invalid,
            category_summary=category_summary,
            execution_statistics=execution_statistics,
            judgements=judgements_text,
        )
        try:
            batch_summary = await complete(
                RoleRequest(
                    system_prompt=self.summarizer_system,
                    user_content=summary_user,
                    temperature=1.0,
                    response_format="text",
                )
            )
        except (RuntimeError, ValueError) as exc:
            batch_summary = (
                f"The candidate scored {n_correct}/{n_scored}; {n_invalid} trajectories were "
                f"environment-invalid. Batch summarization failed: {exc}."
            )
        return _ESIBenchGradient(
            evaluation=evaluation,
            judged=tuple(judged),
            batch_summary=batch_summary,
            n_correct=n_correct,
            n_scored=n_scored,
            n_invalid=n_invalid,
            category_summary=category_summary,
            execution_statistics=execution_statistics,
        )

    @staticmethod
    def _representative(gradient: _ESIBenchGradient, limit: int = 6) -> list[_JudgedTrace]:
        valid = [item for item in gradient.judged if not item.trace.metadata.environment_invalid]
        selected: list[_JudgedTrace] = []
        seen_failures: set[str] = set()
        for item in valid:
            failure = str(item.judgement.get("primary_failure", "other"))
            if not _official_correct(item.trace) and failure not in seen_failures:
                selected.append(item)
                seen_failures.add(failure)
            if len(selected) >= limit:
                return selected
        for item in valid:
            if item not in selected:
                selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @classmethod
    def _task_interfaces(cls, gradient: _ESIBenchGradient) -> str:
        values: list[dict[str, Any]] = []
        for item in cls._representative(gradient):
            extra = _metadata(item.trace)
            values.append(
                {
                    "task_family": _family(item.trace),
                    "task_subfamily": extra.get("task_subfamily"),
                    "question": extra.get("question"),
                    "answer_options": extra.get("options"),
                    "reference_image_count": len(_reference_images(item.trace)),
                    "official_task_contract": extra.get("task_contract"),
                }
            )
        return _json_text(values)

    @classmethod
    def _reports(cls, gradient: _ESIBenchGradient) -> str:
        return "\n\n".join(_json_text(item.judgement) for item in cls._representative(gradient))

    @classmethod
    def _trace_excerpts(cls, gradient: _ESIBenchGradient) -> str:
        excerpts: list[dict[str, Any]] = []
        for item in cls._representative(gradient):
            indices = _selected_round_indices(item.trace.rounds, limit=5)
            excerpts.append(
                {
                    "task_family": _family(item.trace),
                    "official_correct": _official_correct(item.trace),
                    "termination": item.trace.metadata.termination_reason,
                    "steps": [
                        {
                            "step": index + 1,
                            "reason_action": item.trace.rounds[index].planner_response,
                            "parsed_action": item.trace.rounds[index].command,
                            "action_result": item.trace.rounds[index].action_result,
                        }
                        for index in indices
                    ],
                }
            )
        return _json_text(excerpts)

    @classmethod
    def _context_payloads(cls, gradient: _ESIBenchGradient) -> tuple[str, list[dict[str, Any]]]:
        excerpts: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        for item in cls._representative(gradient, limit=4):
            indices = _selected_round_indices(item.trace.rounds, limit=3)
            steps: list[dict[str, Any]] = []
            for index in indices:
                record = item.trace.rounds[index]
                payload_text, payload_images = _split_images(record.context_payload)
                steps.append(
                    {
                        "step": index + 1,
                        "observable_harness_input": record.harness_input,
                        "actual_ordered_context_payload": payload_text,
                        "payload_image_count": len(payload_images),
                        "planner_action": record.command,
                    }
                )
                images.extend(payload_images)
            excerpts.append(
                {
                    "task_family": _family(item.trace),
                    "official_correct": _official_correct(item.trace),
                    "selected_steps": steps,
                }
            )
        return _json_text(excerpts), _deduplicate_images(images)

    @staticmethod
    def _history(context: OptimizerRequestContext) -> str:
        events = [event.model_dump(mode="json") for event in context.optimization_history[-30:]]
        return _json_text(events)

    @staticmethod
    def _artifact_feedback(context: OptimizerRequestContext) -> str:
        recent_errors = [
            event.validation_error for event in context.optimization_history[-30:] if event.validation_error
        ]
        base = "The current artifact passed static validation and all configured smoke probes."
        if recent_errors:
            base += "\nRecent rejected-candidate feedback:\n- " + "\n- ".join(recent_errors[-8:])
        if context.validation_feedback:
            base += context.validation_feedback
        return base

    @classmethod
    def _optimizer_images(cls, gradient: _ESIBenchGradient, stage: str) -> list[dict[str, Any]]:
        images: list[dict[str, Any]]
        if stage == "harness":
            _, images = cls._context_payloads(gradient)
            label = "ACTUAL IMAGES FROM THE REPRESENTATIVE CONTEXT PAYLOADS"
        else:
            images = []
            for item in cls._representative(gradient, limit=4):
                images.extend(_reference_images(item.trace))
                if item.trace.rounds:
                    images.extend(_image_parts(item.trace.rounds[0].observation_before))
                    images.extend(_image_parts(item.trace.rounds[-1].observation_after))
            label = "REPRESENTATIVE OFFICIAL VISUAL EVIDENCE"
        unique = _deduplicate_images(images)
        return [{"type": "text", "text": label}, *unique] if unique else []

    def build_optimizer_request(self, context: OptimizerRequestContext) -> RoleRequest:
        gradient = cast(_ESIBenchGradient, context.textual_gradient)
        common = {
            "candidate_name": gradient.evaluation.candidate_version,
            "n_correct": gradient.n_correct,
            "n_scored": gradient.n_scored,
            "score": f"{gradient.n_correct / max(1, gradient.n_scored):.6f}",
            "n_valid": gradient.n_scored,
            "n_environment_invalid": gradient.n_invalid,
            "batch_summary": gradient.batch_summary,
            "representative_task_interfaces": self._task_interfaces(gradient),
            "representative_judger_reports": self._reports(gradient),
            "representative_trace_excerpts": self._trace_excerpts(gradient),
            "optimization_history": self._history(context),
        }
        if context.stage == "skill":
            user_text = self.skill_optimizer_user.format(
                current_skill=context.parent.skill.template,
                frozen_harness=context.parent.harness.template,
                **common,
            )
            system_prompt = self.skill_optimizer_system
        else:
            context_payloads, _ = self._context_payloads(gradient)
            user_text = self.harness_optimizer_user.format(
                frozen_planner_skill=context.parent.skill.template,
                current_context_code=context.parent.harness.template,
                representative_context_payloads=context_payloads,
                artifact_validation_feedback=self._artifact_feedback(context),
                **common,
            )
            user_text += (
                "\n\nAGENT LIGHTNING HARNESS VALIDATION CONTRACT\n"
                "===========================================\n"
                + context.harness_contract
                + f"\nFunction: {context.harness_function_name}"
                + f"\nSmoke arguments: {list(context.harness_smoke_args)!r}"
            )
            system_prompt = self.harness_optimizer_system + "\n\n" + _AGL_HARNESS_ADAPTER
        if context.validation_feedback and context.stage == "skill":
            user_text += context.validation_feedback
        user_text += f"\n\nProposal round: {context.round_index}; branch: {context.branch_index}."
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(self._optimizer_images(gradient, context.stage))
        return RoleRequest(
            system_prompt=system_prompt,
            user_content=content,
            temperature=1.0,
            response_format="json_object",
        )

    def parse_optimizer_response(self, stage: OptimizationStage, response: str) -> ArtifactProposal:
        del stage
        payload = parse_json_object(response)
        rationale = payload.get("rationale")
        artifact = payload.get("new_artifact")
        if not isinstance(rationale, str) or not isinstance(artifact, str) or not artifact.strip():
            raise ValueError("ESI-Bench optimizer must return string fields rationale and new_artifact.")
        return ArtifactProposal(rationale=rationale, artifact=artifact.strip())


__all__ = ["ESIBenchRoleProtocol"]
