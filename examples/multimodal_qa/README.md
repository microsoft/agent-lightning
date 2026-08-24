# Multimodal QA

A minimal end-to-end example of **multimodal (image) reinforcement learning** with
Agent Lightning + VERL. The task is fully synthetic — no dataset download needed:

- Each sample is a 256x256 image with 1–5 non-overlapping red circles drawn at
  random positions (generated with PIL at startup).
- The agent sends the image (as a base64 `image_url` content part) plus the
  question "How many red circles are in the image?" to the proxied VLM.
- A rule-based reward scores the answer: 1.0 if the first integer in the reply
  equals the true circle count, 0.0 otherwise.

Default model: `Qwen/Qwen2.5-VL-3B-Instruct` (any mrope VLM supported by verl
works; pass `--model` or set `MODEL` to change it).

## Why this example exists

It exercises the multimodal training data path: image URLs are recovered from the
raw `model_request` events, aligned with the training triplets, and turned into
per-row `multi_modal_inputs` (`pixel_values` + `image_grid_thw`) with mrope
`(batch, 4, seq_len)` position ids, so the training forward pass actually
consumes the image features. Multimodal training requires
`agentlightning.trace_aggregator.level: transition` (already set in
`train_multimodal_qa.py`); the adapter raises a clear error if image-bearing
traces are aggregated at the `trajectory` level instead.

## How to run

Requirements: a GPU environment with the VERL extra installed
(`pip install agentlightning[verl]`), plus `openai` and `Pillow`.

```bash
cd examples/multimodal_qa
bash run_local.sh
```

`run_local.sh` starts three components (same layout as the other examples):

1. `agl-server` — the Agent Lightning server with the OpenAI proxy pointing at
   the configured model;
2. `agl-controller` — the local runner that executes `MultimodalQAAgent`
   (defined in `multimodal_qa_agent.py`) per rollout;
3. `train_multimodal_qa.py` — generates the synthetic dataset, builds the VERL
   config, and launches GRPO training.

Useful overrides (passed through to the VERL config, hydra-style):

```bash
bash run_local.sh --train-size 32 --val-size 8 trainer.total_epochs=1
```

## What to expect

- Rollout replies converge towards the correct count and `training/reward`
  climbs from chance level (~0.2) towards 1.0.
- At each training step the batch carries `multi_modal_inputs` in
  `non_tensor_batch` and `position_ids` with `dim() == 3` (mrope). To see the
  vision tensors reach the model forward, log `model_inputs` keys in verl's
  `prepare_model_inputs` — `pixel_values` and `image_grid_thw` should be present.
- Without the multimodal data path (or with images silently dropped), training
  still runs but the warning
  `rollout traces contain images but RolloutAdapter has no processor` appears and
  the vision signal never reaches the training forward.
