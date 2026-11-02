# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import math
from functools import wraps
from itertools import repeat
from typing import Any

import numpy as np
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def gather_score_centering(logprobs, num_logprobs, token_ids):
    from vllm.v1.sample.sampler import Sampler  # pyright: ignore[reportMissingImports]

    return Sampler.gather_logprobs(logprobs, num_logprobs, token_ids)


class ScoreCenteringWorkerExtension(vLLMColocateWorkerExtension):
    model_runner: Any

    def monkey_patch_model(self, vocab_size: int):
        super().monkey_patch_model(vocab_size)
        sampler = self.model_runner.sampler
        if sampler.logprobs_mode != "processed_logprobs":
            raise ValueError("Score centering data requires processed_logprobs")
        sampler.gather_logprobs = gather_score_centering


def extract_score_centering(output, top_logprobs_num=128):
    ids, probabilities = [], []
    if output.logprobs is None or len(output.logprobs) != len(output.token_ids):
        raise ValueError("Missing per-token score centering data")
    for candidates in output.logprobs:
        head = sorted(
            ((int(key), float(value.logprob)) for key, value in candidates.items() if math.isfinite(value.logprob)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_logprobs_num]
        ids.append([key for key, _ in head])
        probabilities.append([value for _, value in head])
    return ids, probabilities


def install_server_support():
    """Install only in the Ray job serving the SC example; preserve ordinary jobs."""
    # vLLM is supplied by the inference environment and is absent from CPU CI.
    from vllm.entrypoints.openai.chat_completion.protocol import (  # pyright: ignore[reportMissingImports]
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.chat_completion.serving import (  # pyright: ignore[reportMissingImports]
        OpenAIServingChat,
    )
    from vllm.logprobs import append_logprobs_for_next_position  # pyright: ignore[reportMissingImports]
    from vllm.v1.engine.logprobs import LogprobsProcessor  # pyright: ignore[reportMissingImports]

    original_update = LogprobsProcessor._update_sample_logprobs
    if getattr(original_update, "_agl_score_centering", False):
        return

    @wraps(original_update)
    def update(self, logprobs_lists):
        for ids, probabilities, rank in zip(*logprobs_lists[:3], strict=True):
            keep = np.isfinite(probabilities)
            self.cumulative_logprob += float(probabilities[0])
            append_logprobs_for_next_position(
                self.logprobs,
                ids[keep].tolist(),
                probabilities[keep].tolist(),
                repeat(None),
                int(rank),
                int(keep.sum()) - 1,
            )

    update._agl_score_centering = True  # pyright: ignore[reportAttributeAccessIssue]
    LogprobsProcessor._update_sample_logprobs = update
    original_full = OpenAIServingChat.chat_completion_full_generator

    @wraps(original_full)
    async def full(self, request, result_generator, *args, **kwargs):
        if not getattr(request, "return_score_centering", False):
            return await original_full(self, request, result_generator, *args, **kwargs)
        if (
            request.n != 1
            or request.temperature <= 0
            or not request.logprobs
            or request.top_logprobs is None
            or request.top_logprobs < 1
        ):
            raise ValueError("SC capture requires n=1, positive temperature and top_logprobs")
        captured: list[Any] = [None]

        async def capture():
            async for result in result_generator:
                captured[0] = result
                yield result

        response = await original_full(self, request.model_copy(update={"top_logprobs": 0}), capture(), *args, **kwargs)
        final = captured[0]
        if final is not None and isinstance(response, ChatCompletionResponse):
            by_index = {output.index: output for output in final.outputs}
            for index, choice in enumerate(response.choices):
                ids, probabilities = extract_score_centering(by_index[choice.index], request.top_logprobs)
                response.choices[index] = choice.model_copy(
                    update={
                        "score_centering_token_ids": ids,
                        "score_centering_log_probs": probabilities,
                    }
                )
        return response

    OpenAIServingChat.chat_completion_full_generator = full
