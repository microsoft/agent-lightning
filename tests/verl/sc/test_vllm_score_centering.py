# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import math
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.workers.utils import padding
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from agentlightning.server.proxy import ProxyRouter
from agentlightning.server.routes.events import _trim_model_request
from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.rollout_adapter import RolloutAdapter
from agentlightning.verl.sc.score_centering import ScoreCenteringConfig, prepare_support_outputs, score_centering_loss


@pytest.fixture(autouse=True)
def padding_without_flash_attention(monkeypatch):
    """Keep VERL packing coverage in the CPU CI environment without FlashAttention."""
    if find_spec("flash_attn") is not None:
        return

    def unpad_input(hidden_states, attention_mask):
        lengths = attention_mask.sum(-1, dtype=torch.int32)
        indices = attention_mask.flatten().nonzero(as_tuple=True)[0]
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        values = hidden_states.flatten(0, 1).index_select(0, indices)
        return values, indices, offsets, int(lengths.max())

    monkeypatch.setattr(padding, "unpad_input", unpad_input)
    monkeypatch.setattr(padding, "index_first_axis", lambda values, indices: values.index_select(0, indices))


def turn(prompt, tokens, support):
    probabilities = [[-math.log(len(ids))] * len(ids) for ids in support]
    trimmed = _trim_model_request(
        {
            "response": {
                "prompt_token_ids": prompt,
                "choices": [
                    {
                        "token_ids": tokens,
                        "logprobs": {"content": [{"logprob": p[0]} for p in probabilities]},
                        "score_centering_token_ids": support,
                        "score_centering_log_probs": probabilities,
                    }
                ],
            }
        }
    )
    return Triplet(
        prompt={"token_ids": trimmed["prompt_token_ids"]},
        response={
            "token_ids": trimmed["response_token_ids"],
            "log_probs": trimmed["response_log_probs"],
            "score_centering_token_ids": trimmed["score_centering_token_ids"],
            "score_centering_log_probs": trimmed["score_centering_log_probs"],
        },
    )


def build_batch(max_response_length=6):
    rollouts = [
        CompletedRollout(
            rollout_id="r1",
            data_id="q1",
            step=0,
            sample_idx_in_step=0,
            enqueue_time=0,
            final_reward=1,
            triplets=[turn([6, 7], [1, 2], [[1, 4], [2, 3, 5]]), turn([6, 7, 1, 2, 6], [3, 4], [[3, 5], [1, 4, 5]])],
        ),
        CompletedRollout(
            rollout_id="r2",
            data_id="q1",
            step=0,
            sample_idx_in_step=1,
            enqueue_time=0,
            final_reward=0,
            triplets=[turn([7], [5], [[1, 5]])],
        ),
    ]
    adapter = RolloutAdapter(
        max_prompt_length=3,
        max_response_length=max_response_length,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level="trajectory",
        require_score_centering=True,
    )
    return adapter.get_train_data_batch(rollouts)[0]


def test_support_survives_tools_truncation_rank_and_microbatch_slicing():
    batch = build_batch(max_response_length=4)
    assert batch.batch["response_mask"].tolist() == [[1, 1, 0, 1], [1, 0, 0, 0]]
    first = batch.non_tensor_batch["score_centering"][0]
    assert first["token_ids"] == [[1, 4], [2, 3, 5], [], [3, 5]]
    reordered = cast(DataProto, batch[[1, 0]]).to_tensordict()
    packed = left_right_2_no_padding(reordered)
    # NonTensorStack follows the same sample selection as the token tensors.
    slices = tu.chunk_tensordict(packed, 2)
    assert tu.get(slices[0], "score_centering")[0]["token_ids"] == [[1, 5]]
    assert tu.get(slices[1], "score_centering")[0] == first


@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_prediction_positions_and_tool_mask_have_correct_gradients(temperature, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    batch = build_batch()
    data = left_right_2_no_padding(batch.to_tensordict()).to(device)
    data["advantages"] = torch.tensor([[0.5] * 6, [-0.5] * 6], device=device)
    tu.assign_non_tensor(data, batch_num_tokens=5, dp_size=1, use_remove_padding=True, use_fused_kernels=False)
    dtype = torch.float32 if device == "cuda" else torch.float64
    logits = torch.arange(9 * 8, dtype=dtype, device=device).reshape(1, 9, 8).div(30).requires_grad_()
    config = ScoreCenteringConfig()
    engine = SimpleNamespace(
        use_ulysses_sp=False,
        score_centering_config=config,
        score_centering_temperature=temperature,
        score_centering_entropy=False,
    )
    output = prepare_support_outputs(engine, SimpleNamespace(logits=logits), {}, data, None)
    sampled_logp = no_padding_2_padding(output["log_probs"], data)
    # Packed rows: [6,7,1,2,6,3,4] and [7,5]; active prediction positions 1,2,4,5,7.
    positions, candidates, targets = [1, 2, 4, 5, 7], [[1, 4], [2, 3, 5], [3, 5], [1, 4, 5], [1, 5]], [1, 2, 3, 4, 5]
    for value, pos, _ids, target in zip(
        sampled_logp[data["response_mask"].bool()], positions, candidates, targets, strict=True
    ):
        torch.testing.assert_close(value, (logits[0, pos] / temperature).log_softmax(0)[target])
    loss, _ = score_centering_loss(config, output, data)
    gradient = torch.autograd.grad(loss, logits)[0][0]
    assert torch.equal(gradient[[0, 3, 6, 8]], torch.zeros_like(gradient[[0, 3, 6, 8]]))
    for pos, ids, target, advantage in zip(positions, candidates, targets, [0.5, 0.5, 0.5, 0.5, -0.5], strict=True):
        p = (logits[0, pos].detach() / temperature).softmax(0)
        rho = 1e-6 / (1 - p[ids].sum())
        correction = torch.zeros(8, dtype=dtype, device=device)
        correction[ids] = 1 / len(ids) - rho * p[ids]
        sampled = torch.eye(8, dtype=dtype, device=device)[target]
        expected = -advantage / 5 / temperature * (sampled - p - correction + correction.sum() * p)
        torch.testing.assert_close(gradient[pos], expected, atol=2e-8, rtol=1e-6)


def test_proxy_shares_top_p_and_only_captures_training_support():
    router = ProxyRouter(
        {
            "model_name": "Qwen/Qwen3-8B",
            "score_centering": {"enabled": True, "top_logprobs_num": 32},
            "train": {"temperature": 1, "top_p": 0.95, "top_k": -1},
            "val": {"temperature": 1, "top_p": 0.95, "top_k": -1},
        }
    )
    train, val = router.prepare_body({"top_p": 0.1}, "train"), router.prepare_body({}, "val")
    assert train["top_p"] == val["top_p"] == 0.95
    assert train["return_score_centering"] and train["logprobs"]
    assert train["top_logprobs"] == 32 and train["top_k"] == -1
    assert "return_score_centering" not in val


def test_vllm_exports_top_128_without_renormalization(monkeypatch):
    pytest.importorskip("vllm")
    import vllm.v1.sample.sampler as sampler_module  # pyright: ignore[reportMissingImports]
    from vllm.v1.engine.logprobs import LogprobsProcessor  # pyright: ignore[reportMissingImports]
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler  # pyright: ignore[reportMissingImports]

    from agentlightning.verl.sc.vllm_score_centering import (
        extract_score_centering,
        gather_score_centering,
        install_server_support,
    )

    # Keep this CPU test independent of the compiled GPU rank-count kernel.
    monkeypatch.setattr(sampler_module, "batched_count_greater_than", lambda a, b: (a > b).sum(-1))
    logits = torch.stack((torch.linspace(-1, 1, 300), torch.arange(300).float()))
    sampler = TopKTopPSampler("processed_logprobs")
    # Use per-row generators so the native CPU path needs no compiled sampler.
    generators = {i: torch.Generator().manual_seed(i) for i in range(2)}
    sampled, log_q = sampler.forward_cpu(logits.clone(), generators, None, torch.tensor([0.95, 0.95]))
    assert log_q is not None
    sampled[0] = torch.where(log_q[0].isfinite())[0][0]
    gathered = gather_score_centering(log_q, 128, sampled.long()).tolists()
    assert torch.isfinite(log_q[0]).sum() > 128
    install_server_support()
    for row in range(2):

        def unexpected_decode(*args):
            pytest.fail("Score centering data must keep integer IDs without decoding candidates")

        tokenizer = SimpleNamespace(decode=unexpected_decode)
        processor = LogprobsProcessor(
            tokenizer=cast(Any, tokenizer),
            logprobs=[],
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=0,
            num_prompt_logprobs=None,
        )
        processor._update_sample_logprobs(gathered.slice_request(row, 1))
        assert processor.tokenizer is tokenizer
        assert processor.logprobs is not None
        assert all(math.isfinite(value.logprob) for value in processor.logprobs[0].values())
        ids, probabilities = extract_score_centering(
            SimpleNamespace(token_ids=[sampled[row].item()], logprobs=processor.logprobs)
        )
        expected = log_q[row].topk(128).indices
        expected = expected[log_q[row, expected].isfinite()]
        assert set(ids[0]) == set(expected.tolist())
        torch.testing.assert_close(torch.tensor(probabilities[0]), log_q[row, ids[0]])
        assert processor.logprobs[0][int(sampled[row].item())].logprob == pytest.approx(log_q[row, sampled[row]].item())
        if row == 0:
            assert sampled[row].item() not in ids[0]
            assert sum(map(math.exp, probabilities[0])) < 1


def test_sc_regularizers_are_separate_masked_token_means():
    from verl.trainer.ppo.core_algos import kl_penalty

    data = left_right_2_no_padding(build_batch().to_tensordict())
    data["advantages"] = torch.tensor([[0.5] * 6, [-0.5] * 6])
    data["ref_log_prob"] = torch.full((2, 6), -2.0, dtype=torch.float64)
    tu.assign_non_tensor(data, batch_num_tokens=5, dp_size=1, use_remove_padding=True, use_fused_kernels=False)
    logits = torch.linspace(-1, 1, 72, dtype=torch.float64).reshape(1, 9, 8).requires_grad_()
    config = ScoreCenteringConfig()
    engine = SimpleNamespace(
        use_ulysses_sp=False,
        score_centering_config=config,
        score_centering_temperature=0.7,
        score_centering_entropy=True,
        score_centering_entropy_coeff=0.1,
    )
    outputs = prepare_support_outputs(engine, SimpleNamespace(logits=logits), {}, data, None)
    plain = {key: value for key, value in outputs.items() if key != "entropy"}
    base, _ = score_centering_loss(config, plain, data)
    # Entropy can be reported without enabling an actor regularizer.
    unregularized, entropy_metrics = score_centering_loss(config, outputs, data)
    torch.testing.assert_close(unregularized, base)
    assert "entropy_loss" in entropy_metrics
    actor = SimpleNamespace(entropy_coeff=0.1, use_kl_loss=True, kl_loss_coef=0.2, kl_loss_type="k3")
    combined, metrics = score_centering_loss(config, outputs, data, actor_config=actor)
    active = data["response_mask"].bool()
    log_p = no_padding_2_padding(outputs["log_probs"], data)
    entropy = no_padding_2_padding(outputs["entropy"], data)
    reference_kl = kl_penalty(logprob=cast(torch.FloatTensor, log_p), ref_logprob=data["ref_log_prob"], kl_penalty="k3")
    expected = base - 0.1 * entropy[active].mean() + 0.2 * reference_kl[active].mean()
    torch.testing.assert_close(combined, expected)
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(combined, logits)[0], expected_grad)
    assert {"pg_loss", "entropy_loss", "kl_loss"} <= metrics.keys()


def test_sc_k3_uses_sampled_probabilities_and_original_token_mask():
    data = left_right_2_no_padding(build_batch().to_tensordict())
    data["advantages"] = torch.ones(2, 6)
    tu.assign_non_tensor(data, batch_num_tokens=5, dp_size=1, use_remove_padding=True, use_fused_kernels=False)
    config = ScoreCenteringConfig(is_mode="mis")
    engine = SimpleNamespace(
        use_ulysses_sp=False,
        score_centering_config=config,
        score_centering_temperature=0.7,
        score_centering_entropy=False,
    )
    logits = torch.linspace(-1, 1, 72, dtype=torch.float64).reshape(1, 9, 8).requires_grad_()
    outputs = prepare_support_outputs(engine, SimpleNamespace(logits=logits), {}, data, None)
    active = data["response_mask"].bool()
    data["rollout_log_probs"] = data["rollout_log_probs"].masked_fill(~active, torch.nan)
    _, metrics = score_centering_loss(config, outputs, data)
    log_p = no_padding_2_padding(outputs["log_probs"], data)[active]
    ratio = log_p.detach().exp() / data["rollout_log_probs"][active].double().exp()
    expected = (ratio - ratio.log() - 1).mean()
    assert metrics["rollout_k3_kl"].aggregate() == pytest.approx(expected.item())


def test_padded_sc_outputs_and_gradients_match_separate_sequences():
    data = left_right_2_no_padding(build_batch().to_tensordict())
    data["advantages"] = torch.tensor([[0.5] * 6, [-0.5] * 6])
    tu.assign_non_tensor(data, batch_num_tokens=5, dp_size=1, use_remove_padding=False, use_fused_kernels=False)
    logits = torch.linspace(-1, 1, 2 * 7 * 8, dtype=torch.float64).reshape(2, 7, 8).requires_grad_()
    config = ScoreCenteringConfig(is_mode="tis")
    engine = SimpleNamespace(
        use_ulysses_sp=False,
        score_centering_config=config,
        score_centering_temperature=0.7,
        score_centering_entropy=False,
    )
    padded = prepare_support_outputs(engine, SimpleNamespace(logits=logits), {}, data, None)
    loss, _ = score_centering_loss(config, padded, data)
    packed_logits = torch.cat((logits[0, :7], logits[1, :2])).unsqueeze(0)
    tu.assign_non_tensor(data, use_remove_padding=True)
    packed = prepare_support_outputs(engine, SimpleNamespace(logits=packed_logits), {}, data, None)
    reference, _ = score_centering_loss(config, packed, data)
    torch.testing.assert_close(loss, reference)
    for key in padded:
        torch.testing.assert_close(padded[key].values(), packed[key].values())
    expected_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.equal(actual_grad[1, 2:], torch.zeros_like(actual_grad[1, 2:]))
