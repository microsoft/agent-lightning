# Copyright (c) Microsoft. All rights reserved.

"""Paper-faithful SHAPER role protocol for VLABench."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

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

_AGL_HARNESS_ADAPTER = """
## Agent Lightning execution adapter (authoritative for this contrib)

The paper prompt above describes the original in-process APCO runtime. This
contrib executes generated harnesses in an isolated JSON-only worker. Preserve
the same optimization objective, but obey the following concrete interface:

- Define exactly `def build_context(history)`.
- `history` is a list of dictionaries. Use `record.get(...)`, not attribute
  access. Available keys are `round_index`, `task_instruction`,
  `planner_response`, `command`, `execution_steps`, `observation_before`,
  `observation_after`, `action_result`, and `runtime_errors`.
- Observations are already OpenAI `text` / `image_url` content parts. Reuse
  useful image parts directly; do not call `encode_image`.
- `llm_client`, `llm_model`, `numpy`, and simulator objects are not available.
  The harness must be deterministic and may not make network calls.
- The validation contract supplied in the user message overrides incompatible
  low-level runtime details in the original prompt.
""".strip()


@dataclass(frozen=True)
class _VLABenchGradient:
    evaluation: CandidateEvaluation
    summaries: tuple[str, ...]
    judgements: tuple[tuple[dict[str, Any], ...], ...]


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


def _content_parts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        return [{"type": "text", "text": json.dumps(value, ensure_ascii=False, default=str)}]
    output: list[dict[str, Any]] = []
    for item in cast(list[Any], value):
        if not isinstance(item, dict):
            continue
        mapping = cast(dict[str, Any], item)
        if mapping.get("type") in {"text", "image_url"}:
            output.append(mapping)
    return output


def _task_instruction(trace: EpisodeTrace) -> str:
    return trace.rounds[0].task_instruction if trace.rounds else ""


def _episode_success(trace: EpisodeTrace) -> bool:
    return float(trace.final_reward or 0.0) >= 1.0


def _clean_markdown_artifact(value: str) -> str:
    cleaned = value.strip()
    fence = re.fullmatch(r"```(?:text|plaintext|python)?\s*\n?(.*?)```", cleaned, flags=re.DOTALL | re.I)
    return fence.group(1).strip() if fence else cleaned


class VLABenchRoleProtocol:
    """Use the paper's VLABench Judger, Summarizer, and optimizer prompts."""

    def __init__(self, prompt_dir: Path) -> None:
        self.judger_system = load_text(prompt_dir, "round_judger.txt")
        self.judger_user = load_text(prompt_dir, "round_judger_user.txt")
        self.summarizer_system = load_text(prompt_dir, "episode_summarizer.txt")
        self.summarizer_user = load_text(prompt_dir, "episode_summarizer_user.txt")
        self.skill_optimizer_system = load_text(prompt_dir, "skill_optimizer.txt")
        self.skill_optimizer_user = load_text(prompt_dir, "skill_optimizer_user.txt")
        self.harness_optimizer_system = load_text(prompt_dir, "harness_optimizer.txt")
        self.harness_optimizer_user = load_text(prompt_dir, "harness_optimizer_user.txt")

    async def _judge_round(self, record: RoundRecord, complete: RoleCompleter) -> dict[str, Any]:
        context_text, _ = _split_images(record.context_payload)
        user_text = self.judger_user.format(
            overall_task=record.task_instruction,
            round_idx=record.round_index + 1,
            subtask=record.command,
            vla_steps=record.execution_steps,
            vlm_reasoning=record.planner_response,
            context_output=json.dumps(context_text, ensure_ascii=False, default=str),
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "text", "text": "BEFORE observation (Main, then Wrist):"},
        ]
        content.extend(record.observation_before)
        content.append({"type": "text", "text": "AFTER observation (Main, then Wrist):"})
        content.extend(record.observation_after)
        context_parts = _content_parts(record.context_payload)
        if context_parts:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Additional multimodal context shown to the planner this round; "
                        "these are not BEFORE/AFTER transition images:"
                    ),
                }
            )
            content.extend(context_parts)
        raw = await complete(
            RoleRequest(
                system_prompt=self.judger_system,
                user_content=content,
                temperature=1.0,
                response_format="json_object",
            )
        )
        payload = parse_json_object(raw)
        status = str(payload.get("subtask_status", "")).lower()
        score = float(payload.get("success_score", -1.0))
        if status not in {"success", "partial", "failed"}:
            raise ValueError(f"VLABench Judger returned invalid subtask_status: {status!r}")
        if score not in {0.0, 0.25, 0.5, 0.75, 1.0}:
            raise ValueError(f"VLABench Judger returned invalid success_score: {score!r}")
        required = (
            "observation_analysis",
            "execution_analysis",
            "reasoning_analysis",
            "failure_causes",
            "improvement_suggestions",
            "context_analysis",
        )
        for key in required:
            if not isinstance(payload.get(key), str):
                raise ValueError(f"VLABench Judger omitted string field {key!r}.")
        payload["subtask_status"] = status
        payload["success_score"] = score
        return payload

    @staticmethod
    def _rounds_detail(trace: EpisodeTrace, judgements: Sequence[dict[str, Any]]) -> str:
        lines: list[str] = []
        last_index = len(trace.rounds) - 1
        for index, record in enumerate(trace.rounds):
            judgement = judgements[index] if index < len(judgements) else {}
            lines.append(
                f'Round {record.round_index + 1}: subtask="{record.command}" | '
                f"VLA steps={record.execution_steps} | "
                f"status={judgement.get('subtask_status', 'unknown')} | "
                f"score={judgement.get('success_score', 0.0)}"
            )
            lines.append("  Judger: " + json.dumps(judgement, ensure_ascii=False, default=str))
            if index == last_index:
                context_text, _ = _split_images(record.context_payload)
                lines.append(
                    "  Final context provided to planner: " + json.dumps(context_text, ensure_ascii=False, default=str)
                )
        return "\n".join(lines) if lines else "(No planner rounds were emitted.)"

    async def _summarize_episode(
        self,
        trace: EpisodeTrace,
        judgements: Sequence[dict[str, Any]],
        complete: RoleCompleter,
    ) -> str:
        result = "SUCCESS" if _episode_success(trace) else "FAILED"
        user_text = self.summarizer_user.format(
            overall_task=_task_instruction(trace),
            episode_result=f"{result} (progress={float(trace.final_reward or 0.0):.2f})",
            total_rounds=len(trace.rounds),
            rounds_detail=self._rounds_detail(trace, judgements),
        )
        return await complete(
            RoleRequest(
                system_prompt=self.summarizer_system,
                user_content=user_text,
                temperature=1.0,
                response_format="text",
            )
        )

    async def _diagnose_episode(
        self,
        trace: EpisodeTrace,
        complete: RoleCompleter,
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        if trace.rounds:
            raw_judgements = await asyncio.gather(
                *(self._judge_round(record, complete) for record in trace.rounds),
                return_exceptions=True,
            )
            judgements: list[dict[str, Any]] = []
            for record, value in zip(trace.rounds, raw_judgements):
                if isinstance(value, BaseException):
                    judgements.append(
                        {
                            "subtask_status": "failed",
                            "success_score": 0.0,
                            "observation_analysis": "Judger output unavailable.",
                            "execution_analysis": "Judger output unavailable.",
                            "reasoning_analysis": "Judger output unavailable.",
                            "failure_causes": f"Diagnostic failure: {value}",
                            "improvement_suggestions": "Do not infer a change from this round alone.",
                            "context_analysis": "Judger output unavailable.",
                            "round_index": record.round_index,
                        }
                    )
                else:
                    judgements.append(value)
        else:
            judgements = []
        try:
            summary = await self._summarize_episode(trace, judgements, complete)
        except (RuntimeError, ValueError) as exc:
            summary = (
                f"The episode ended with reward {float(trace.final_reward or 0.0):.2f} after "
                f"{len(trace.rounds)} planner rounds. Episode summarization failed: {exc}."
            )
        return summary, tuple(judgements)

    async def build_textual_gradient(
        self,
        evaluation: CandidateEvaluation,
        complete: RoleCompleter,
    ) -> _VLABenchGradient:
        traces = [trace for trace in evaluation.traces if not trace.metadata.environment_invalid]
        diagnosed = await asyncio.gather(*(self._diagnose_episode(trace, complete) for trace in traces))
        return _VLABenchGradient(
            evaluation=evaluation,
            summaries=tuple(item[0] for item in diagnosed),
            judgements=tuple(item[1] for item in diagnosed),
        )

    @staticmethod
    def _representative_rounds(
        evaluation: CandidateEvaluation, limit: int = 4
    ) -> list[tuple[EpisodeTrace, RoundRecord]]:
        traces = [trace for trace in evaluation.traces if not trace.metadata.environment_invalid]
        selected: list[tuple[EpisodeTrace, RoundRecord]] = []
        first = next(((trace, trace.rounds[0]) for trace in traces if trace.rounds), None)
        if first is not None:
            selected.append(first)
        failures = [(trace, record) for trace in traces if not _episode_success(trace) for record in trace.rounds[1:]]
        if failures:
            selected.append(max(failures, key=lambda item: item[1].round_index))
        successes = [(trace, record) for trace in traces if _episode_success(trace) for record in trace.rounds[1:]]
        if successes:
            selected.append(max(successes, key=lambda item: item[1].round_index))
        seen = {(trace.rollout_id, record.round_index) for trace, record in selected}
        for trace in traces:
            for record in trace.rounds:
                key = (trace.rollout_id, record.round_index)
                if key not in seen:
                    selected.append((trace, record))
                    seen.add(key)
                if len(selected) >= limit:
                    return selected[:limit]
        return selected[:limit]

    @classmethod
    def _multimodal_traces(cls, evaluation: CandidateEvaluation) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "## Context Code Execution Traces\n"
                    "These show what build_context() actually supplied and how the planner responded."
                ),
            }
        ]
        for index, (trace, record) in enumerate(cls._representative_rounds(evaluation), start=1):
            outcome = "SUCCESS" if _episode_success(trace) else "FAILED"
            blocks.append(
                {
                    "type": "text",
                    "text": (
                        f"### Trace {index}: Round {record.round_index + 1}/{len(trace.rounds)}, "
                        f"episode {outcome}\nbuild_context() output:"
                    ),
                }
            )
            blocks.extend(_content_parts(record.context_payload))
            blocks.append({"type": "text", "text": f'Planner subtask: "{record.command}"'})
        return blocks if len(blocks) > 1 else []

    @staticmethod
    def _history(context: OptimizerRequestContext) -> str:
        events = [event.model_dump(mode="json") for event in context.optimization_history[-30:]]
        return json.dumps(events, ensure_ascii=False, indent=2)

    def build_optimizer_request(self, context: OptimizerRequestContext) -> RoleRequest:
        gradient = cast(_VLABenchGradient, context.textual_gradient)
        traces = [trace for trace in gradient.evaluation.traces if not trace.metadata.environment_invalid]
        successes = sum(_episode_success(trace) for trace in traces)
        average_reward = sum(float(trace.final_reward or 0.0) for trace in traces) / len(traces) if traces else 0.0
        episode_context = (
            f"Episodes: {len(traces)}, Success: {successes}/{len(traces)} "
            f"({100.0 * successes / max(1, len(traces)):.1f}%), Avg reward: {average_reward:.3f}"
        )
        critique = "\n\n".join(
            f"Episode {index}: {summary}" for index, summary in enumerate(gradient.summaries, start=1)
        )
        template = self.skill_optimizer_user if context.stage == "skill" else self.harness_optimizer_user
        user_text = template.format(
            current_prompt=context.parent.skill.template,
            current_context_code=context.parent.harness.template,
            episode_context=episode_context,
            critique=critique,
        )
        user_text = (
            "## Optimization History (recent)\n"
            + self._history(context)
            + "\n\n"
            + user_text
            + f"\n\nProposal round: {context.round_index}; branch: {context.branch_index}."
        )
        system_prompt = self.skill_optimizer_system
        if context.stage == "harness":
            system_prompt = self.harness_optimizer_system + "\n\n" + _AGL_HARNESS_ADAPTER
            user_text += (
                "\n\nAGENT LIGHTNING HARNESS VALIDATION CONTRACT\n"
                "===========================================\n"
                + context.harness_contract
                + f"\nFunction: {context.harness_function_name}"
                + f"\nSmoke arguments: {list(context.harness_smoke_args)!r}"
            )
        if context.validation_feedback:
            user_text += context.validation_feedback
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(self._multimodal_traces(gradient.evaluation))
        return RoleRequest(
            system_prompt=system_prompt,
            user_content=content,
            temperature=1.0,
            response_format="text",
        )

    def parse_optimizer_response(self, stage: OptimizationStage, response: str) -> ArtifactProposal:
        marker = "### Improved Prompt" if stage == "skill" else "### Improved Context Code"
        if marker not in response:
            raise ValueError(f"VLABench optimizer response is missing {marker!r}.")
        prefix, artifact_section = response.split(marker, maxsplit=1)
        analysis = prefix.split("### Analysis", maxsplit=1)[-1].strip()
        artifact = _clean_markdown_artifact(artifact_section)
        if not artifact:
            raise ValueError("VLABench optimizer returned an empty replacement artifact.")
        return ArtifactProposal(rationale=analysis or "No analysis supplied.", artifact=artifact)


__all__ = ["VLABenchRoleProtocol"]
