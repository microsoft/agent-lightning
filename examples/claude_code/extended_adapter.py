# Copyright (c) Microsoft. All rights reserved.

"""Custom adapter module for converting LLM proxy traces to augmented trajectories.

This module provides an augmented LlmProxyTraceToTriplet adapter that converts
LLM proxy spans into augmented trajectories for analysis and evaluation.
It extends the base LlmProxyTraceToTriplet to include additional metadata like chat messages,
log probabilities, and sequence IDs.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

from agentlightning.adapter.triplet import LlmProxyTraceToTriplet
from agentlightning.types import Span, Triplet

logger = logging.getLogger(__name__)


class ExtendedLlmProxyTraceToTriplet(LlmProxyTraceToTriplet):
    """Convert LLM Proxy spans into trajectories with logprobs and customized metadata.

    Augmented fields include:

    - chat messages history from [`llm.hosted_vllm.messages`], saved to `Triplet.metadata['messages']`
    - logprobs from [`llm.hosted_vllm.choices`], saved to `Triplet.response['logprobs']`
    - sequence_id from [`Span.sequence_id`] to locate the order of the span (conversation turn), saved to `Triplet.metadata['sequence_id']`
    """

    def _extract_tokens_from_raw(self, attrs: Dict[str, Any]) -> Tuple[List[int], List[int], List[float]]:  # type: ignore
        """Extract token ids from raw_gen_ai_request attributes.

        - llm.hosted_vllm.prompt_token_ids: string -> List[int]
        - llm.hosted_vllm.choices: string -> [{'token_ids': [...]}] -> take first
        """
        prompt_ids: List[int] = []
        resp_ids: List[int] = []
        logprobs: List[float] = []

        # prompt
        p = attrs.get("llm.hosted_vllm.prompt_token_ids")
        p = self._literal_eval_maybe(p)
        if isinstance(p, list) and all(isinstance(x, int) for x in p):  # type: ignore
            prompt_ids = cast(List[int], p)

        choices = attrs.get("llm.hosted_vllm.choices")
        choices = self._literal_eval_maybe(choices)
        if isinstance(choices, list) and choices:
            cand = cast(Any, choices[0])
            if isinstance(cand, dict):
                tids = cast(Dict[str, Any], cand).get("token_ids")
                if isinstance(tids, list) and all(isinstance(x, int) for x in tids):  # type: ignore
                    resp_ids = cast(List[int], tids)

                if "logprobs" in cand:
                    logprobs_dict = cast(Dict[str, Any], cand).get("logprobs")
                    if isinstance(logprobs_dict, dict) and "content" in logprobs_dict:
                        content = cast(List[Dict[str, Any]], logprobs_dict["content"])
                        logprobs = [float(item["logprob"]) for item in content if "logprob" in item]

        return prompt_ids, resp_ids, logprobs

    def adapt(self, source: List[Span], /) -> List[Triplet]:  # type: ignore
        """Convert LLM Proxy spans into [`Triplet`][agentlightning.Triplet] trajectories.

        Args:
            source: Spans emitted by the LLM Proxy containing prompt, response, and reward data.

        Returns:
            Ordered trajectory transitions matched purely by `sequence_id`.
        """
        # 1) Sort deterministically by (sequence_id, start_time).
        spans = sorted(
            source,
            key=lambda s: (s.sequence_id, s.start_time),
        )

        # 2) Collect LLM calls
        llm_items: List[Dict[str, Any]] = []
        seen_request_ids: set[str] = set()
        for s in spans:
            attrs = s.attributes or {}
            prompt_ids: List[int] = []
            resp_ids: List[int] = []
            logprobs: List[float] = []

            if s.name == "raw_gen_ai_request":
                prompt_ids, resp_ids, logprobs = self._extract_tokens_from_raw(attrs)

                if len(prompt_ids) == 0 or len(resp_ids) == 0:
                    logger.warning(
                        f"Span {s.span_id} is missing prompt (len={len(prompt_ids)}) or response (len={len(resp_ids)}) token ids. Ignoring this span."
                    )
                    continue
                elif len(logprobs) == 0:
                    logger.warning(f"Span {s.span_id} is missing logprobs. Ignoring logprobs for this span.")
                    continue
                elif len(resp_ids) != len(logprobs):
                    logger.warning(
                        f"Span {s.span_id} has mismatched response ids and logprobs lengths: "
                        f"{len(resp_ids)} vs {len(logprobs)}. Ignoring this span."
                    )
                    continue

            if prompt_ids and resp_ids and logprobs:
                rid = self._request_id_from_attrs(attrs)
                if rid:
                    # Duplicated request ID. This request is already handled.
                    if rid in seen_request_ids:
                        continue
                    seen_request_ids.add(rid)
                llm_items.append(
                    dict(
                        span=s,
                        seq=s.sequence_id,
                        response_ids=resp_ids,
                        prompt_ids=prompt_ids,
                        request_id=rid,
                        logprobs=logprobs,
                    )
                )

        # Order LLM items by sequence only.
        llm_items.sort(key=lambda x: x["seq"])

        # Collect rewards by sequence only.
        rewards: List[Tuple[int, Optional[float]]] = []
        for s in spans:
            val = self._maybe_reward_value(s)
            if val is not None:
                rewards.append((s.sequence_id, val))

        # First-occurrence matching by sequence_id only:
        # For reward at sequence R, assign to the most recent unmatched LLM with seq < R.
        assigned: Dict[str, Optional[float]] = {}
        for r_seq, r_val in sorted(rewards, key=lambda x: x[0]):
            for item in reversed(llm_items):
                sid = item["span"].span_id
                if sid in assigned:
                    continue
                if item["seq"] < r_seq:
                    assigned[sid] = r_val
                    break

        # Build triplets in LLM sequence order.
        triplets: List[Triplet] = []
        for item in llm_items:
            s = item["span"]
            triplets.append(
                Triplet(
                    prompt={"token_ids": item["prompt_ids"]},
                    response={"token_ids": item["response_ids"], "logprobs": item["logprobs"]},
                    reward=assigned.get(s.span_id, None),
                    metadata=dict(
                        # This is called response_id to align with the other adapters.
                        response_id=item["request_id"],
                        sequence_id=item["seq"],
                        messages=self._literal_eval_maybe(s.attributes.get("llm.hosted_vllm.messages")),
                    ),
                )
            )

        return triplets
