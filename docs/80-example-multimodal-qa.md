# Multimodal QA

| GPU | Model | Controller Mode | Trainer Mode | Code |
|---|---|---|---|---|
| 1× B200 185GB | `Qwen/Qwen3.5-2B` | Local | Sync only | [Source](https://github.com/microsoft/agent-lightning/tree/main/examples/multimodal_qa) |

Multimodal QA is a minimal end-to-end example of **multimodal (image) reinforcement learning** with `verl` and Agent Lightning >=v1.0. It trains a vision-language model on a fully synthetic single-image QA task, so no dataset download is needed.

The example exercises the multimodal training data path: image URLs are recovered from the raw `model_request` events, aligned with the training triplets, and turned into per-row `multi_modal_inputs` (`pixel_values` + `image_grid_thw`) with mrope `(batch, 4, seq_len)` position ids, so the training forward pass actually consumes the image features.

## Data Preparation

The dataset is generated with PIL at startup — there is nothing to download:

- Each sample is a 256x256 image with 1–5 non-overlapping red circles drawn at random positions.
- The agent sends the image (as a base64 `image_url` content part) plus the question "How many red circles are in the image?" to the proxied VLM.
- A rule-based reward scores the answer: 1.0 if the first integer in the reply equals the true circle count, 0.0 otherwise.

Any mrope VLM supported by verl works; pass `--model` to change the default model.

## Training

Make sure you have activated a GPU environment with the VERL extra installed (`pip install agentlightning[verl]`), plus `openai` and `Pillow`. Then start training:

```bash
cd examples/multimodal_qa
bash run_local.sh
```

`run_local.sh` starts three components (same layout as the other examples):

1. `agl-server` — the Agent Lightning server with the OpenAI proxy pointing at the configured model;
2. `agl-controller` — the local runner that executes `MultimodalQAAgent` (defined in `multimodal_qa_agent.py`) per rollout;
3. `train_multimodal_qa.py` — generates the synthetic dataset, builds the VERL config, and launches GRPO training.

Useful overrides (passed through to the VERL config, hydra-style):

```bash
bash run_local.sh --train-size 32 --val-size 8 trainer.total_epochs=1
```

Multimodal training requires `agentlightning.trace_aggregator.level: transition` (already set in `train_multimodal_qa.py`); the adapter raises a clear error if image-bearing traces are aggregated at the `trajectory` level instead.

## What to Expect

- Rollout replies converge towards the correct count and `training/reward` climbs from chance level (~0.2) towards 1.0.
- At each training step the batch carries `multi_modal_inputs` in `non_tensor_batch` and `position_ids` with `dim() == 3` (mrope). To see the vision tensors reach the model forward, log `model_inputs` keys in verl's `prepare_model_inputs` — `pixel_values` and `image_grid_thw` should be present.
- Without the multimodal data path (or with images silently dropped), training still runs but the warning `rollout traces contain images but RolloutAdapter has no processor` appears and the vision signal never reaches the training forward.

## Troubleshooting

- **`AssertionError: Expected a cached item for mm_hash=...`, or rollouts hanging after a sleep/wake cycle**: on vLLM < 0.22.0 the prefix cache can desync from the multimodal receiver cache across the trainer's sleep/wake cycles ([vllm#42995](https://github.com/vllm-project/vllm/issues/42995), fixed upstream by [vllm#43001](https://github.com/vllm-project/vllm/pull/43001)). This is a vLLM version issue, not an Agent Lightning bug. On affected versions, keep `actor_rollout_ref.rollout.enable_prefix_caching: False` (the default in this example) and restart the run — resuming from the latest checkpoint if you have checkpointing enabled.
