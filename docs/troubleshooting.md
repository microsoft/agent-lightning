# Troubleshooting

## Multi-GPU Rollout Low Utilization

### Problem
When running rollouts on multi-GPU setups, you may observe that only one GPU is actively utilized while others remain idle, leading to significantly longer rollout times.

### Root Cause
By default, Agent Lightning may launch a limited number of actors during the rollout phase. With insufficient actor parallelism, the workload cannot be effectively distributed across all available GPUs.

### Solution

#### 1. Increase Actor Count
Configure the number of actors to match or exceed the number of available GPUs. For example, if you have 8 GPUs, consider launching at least 8 actors (or more) to ensure all GPUs are utilized:

```python
# In your trainer configuration
trainer = Trainer(
    num_actors=8,  # Adjust based on your GPU count
    # ... other configurations
)
```

#### 2. Check Resource Allocation
Ensure that:
- Each actor is assigned to a different GPU
- GPU memory is sufficient to run multiple actors
- System resources (CPU, RAM) are not bottlenecking actor spawning

#### 3. Monitor GPU Utilization
Use tools like `nvidia-smi` to verify that all GPUs are actively processing:

```bash
watch -n 1 nvidia-smi
```

### Best Practices
- Start with `num_actors = num_gpus` and increase if needed
- For larger models, you may need to reduce actors per GPU to avoid OOM errors
- Consider batch size and model size when determining optimal actor count

### Related Issues
- [#82: Rollout phase takes lots of time, and I find that only 1 of 8 GPU is working when rollout](https://github.com/microsoft/agent-lightning/issues/82)
- [#104: I find agent lightning only use 1 actor during rollout, can we launch multiple actors?](https://github.com/microsoft/agent-lightning/issues/104)

## Response Mask Contiguity

### Problem
You may encounter errors or unexpected behavior related to non-contiguous `response_mask` tensors during training.

### Root Cause
When `compute_response_mask` is called in certain parts of the pipeline (e.g., in `trainer.py`), the resulting mask may not be stored in contiguous memory, which can cause issues with downstream operations.

### Solution

#### Option 1: Ensure Contiguity
If you're implementing custom response mask computation, always ensure the tensor is contiguous:

```python
response_mask = compute_response_mask(...)  # Your computation
response_mask = response_mask.contiguous()  # Ensure contiguity
```

#### Option 2: Move Computation to Appropriate Location
Consider computing the response mask in the data loading or preprocessing stage (e.g., in `daemon.py`) rather than during training, to ensure masks are properly formatted before reaching the trainer.

### Error Example
If you see errors like:
```
RuntimeError: Expected a contiguous tensor for response_mask
```

This indicates the mask needs to be made contiguous before use.

### Prevention
To catch this issue early, you can add a validation check:

```python
def validate_response_mask(response_mask):
    if not response_mask.is_contiguous():
        raise ValueError(
            "response_mask must be contiguous. "
            "Call .contiguous() on the mask before passing to the trainer. "
            "Consider computing response_mask earlier in the pipeline (e.g., in daemon.py) "
            "to ensure proper memory layout."
        )
    return response_mask
```

### Related Issues
- [#119: I think dont put compute_response_mask in trainer.py, otherwise response_mask may not contiguous, put it in deamon.py?](https://github.com/microsoft/agent-lightning/issues/119)

## Additional Resources

For more troubleshooting tips and community discussions, visit:
- [GitHub Issues](https://github.com/microsoft/agent-lightning/issues)
- [Discord Community](https://discord.gg/RYk7CdvDR7)
- [Documentation](https://microsoft.github.io/agent-lightning/)
