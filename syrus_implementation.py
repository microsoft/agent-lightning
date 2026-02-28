```python
# Extract ground truth from each batch item, handling missing keys gracefully
sample_gts = [
    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
    for item in batch
]

# Persist generations with associated metadata for analysis
self._dump_generations(
    inputs=inputs,
    outputs=outputs,
    scores=scores,
    gts=sample_gts,
    reward_extra_infos_dict=reward_extra_infos_dict,
    dump_path=rollout_data_dir,
)
```