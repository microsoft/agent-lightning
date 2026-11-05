# Sealed process-credit arithmetic

This three-call agent uses deterministic integer checks and native process-credit clients. See [the full protocol, estimator and supported training boundary](../../docs/36-process-credit.md). The example demonstrates integration; no learning improvement or GPU training result is claimed.

Run the normal Agent Lightning server and local controller with matching `AGL_KEY`/`AGL_BASE_URL`, following the [quick start](../../docs/01-quick-start.md). Install the native GPU training stack with **VERL 0.8.0**, provide a local model with zero dropout, and run from the repository root:

```bash
python -m examples.process_credit.train --model /path/to/model --print-config
python -m examples.process_credit.train --model /path/to/model
```

The first command validates/prints configuration without loading model weights. The second trains through the existing `run_ppo` entrypoint and performs real generation. Keep serving/training weights and effective sampling distributions identical. The CPU dependency group is for tests/type checking and cannot run GPU training.

`rollout.n=2` fixes two independent episodes per task for whole-task leave-one-out. The local environment maps `a,b,c,d` task fields into the agent. The first integer checker uses only action 0, the delayed checker judges action 0 using action 1 evidence, and the final checker emits a separate unweighted outcome. Stable UUID5 reward IDs and logical request IDs support retrying the same delivery. Ambiguous uncaptured failures abort instead of silently training an incomplete trace.

Audit artifacts default to `outputs/process-credit/process_credit`; checkpoints record the last completed update/config digest. Offline replay does not rerun the model or resume an interrupted actor update. The tiny in-memory train/validation dataset is an integration example, not a held-out task evaluation.
