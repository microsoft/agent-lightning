# Coding Agent: MoE

| GPU | Model | Actor | Rollout | Router replay |
|---|---|---|---|---|
| 4× B200 | `Qwen/Qwen3.5-35B-A3B` | Megatron | vLLM | R3 |

This is the MoE variant of the [Coding Agent](75-example-coding-agent.md) example. It reuses the same
SWE-smith data, Kubernetes controller, repository images, agent, and reward. The existing
`Qwen/Qwen3.5-9B` FSDP path remains available through `examples/swe_smith/run.sh`.

## Results

With pure RL, Qwen3.5-35B-A3B improves from 47.8% to 61.6% on SWE-bench Verified after
1,792 training examples (about 1.8K), a gain of 13.8 percentage points. Results use the
official SWE-bench Verified harness on all 500 instances; trained example counts assume
16 examples per step.

| Step | Trained examples | Result on SWE-bench Verified |
|---|---:|---:|
| 0 (base) | 0 | 47.8% (239/500) |
| 64 | 1,024 | 58.8% (294/500) |
| 112 | 1,792 | 61.6% (308/500) |
| 176 | 2,816 | 60.6% (303/500) |

## Why R3

An MoE token can select different experts during rollout and training. R3 records vLLM's rollout
routing and passes it through the Agent Lightning event, triplet, and `DataProto` pipeline so the
Megatron actor update replays the same experts.

The MoE launcher enables both sides:

```text
actor_rollout_ref.rollout.enable_rollout_routing_replay=true
actor_rollout_ref.actor.megatron.router_replay.mode=R3
```

## Run

Use the MoE wrapper for all three roles:

```bash
# Machine B
export AGL_SERVER_PUBLIC_HOST=<address-reachable-from-controller-and-pods>
export AGL_KEY=<shared-secret>
examples/swe_smith/run_moe.sh server

# Machine A
export AGL_SERVER_PUBLIC_HOST=<gateway-address>
export AGL_KEY=<same-shared-secret>
export AGL_NAMESPACE=agents
examples/swe_smith/run_moe.sh controller

# Machine B
export AGL_KEY=<same-shared-secret>
examples/swe_smith/run_moe.sh trainer
```

`run_moe.sh` selects `train_smith_agent_moe.py`, requests routed experts from the Gateway, and keeps
the standard 9B launcher unchanged. R3 route compaction requires the default trajectory aggregator;
do not override it to `transition`. Other additional arguments are VERL dotlist overrides:

```bash
examples/swe_smith/run_moe.sh trainer \
    trainer.total_training_steps=100 \
    actor_rollout_ref.rollout.n=4
```

The default topology is PP=1, TP=1, EP=4, and ETP=1 with parameter, optimizer, and gradient offload.
Treat it as a four-B200 starting point and tune batch sizes for the available memory.
