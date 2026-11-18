# Basics

Agent Lightning v1.0 consists of three main components: the **API Gateway**, the **Rollout Controller**, and the **Customized Trainer**. Together, they connect existing agents to reinforcement learning through an OpenAI-compatible endpoint, without requiring changes to the agent's interaction loop.

## Overview

<p align="center">
	<img src="../images/architecture.jpg" alt="Agent Lightning v1.0 architecture" width="75%">
</p>

The API Gateway stores the core rollouts, model endpoints, and events and provides an OpenAI-compatible model proxy for agent requests. The Rollout Controller launches and manages agent executions as local processes or Kubernetes Jobs. Finally, the Customized Trainer runs model inference and optimization on the GPU side and turns the rollout data collected by the Gateway into policy updates.

This separation provides several practical advantages:

- **Zero-code-change agent integration:** an existing agent connects by redirecting its OpenAI-compatible model endpoint to the Gateway.
- **Independent resources:** model training and agent execution can run on separate machines or clusters and scale independently.
- **Open infrastructure:** agents can run on a self-hosted Kubernetes cluster instead of requiring a commercial sandbox service.

Below, we briefly introduce each component.

## API Gateway

The API Gateway is a lightweight service at the center of Agent Lightning. It stores rollouts, model endpoints, and events. It also provides an OpenAI-compatible proxy for agents to access model inference.

<p align="center">
	<img src="../images/agentlightning-schema.jpg" alt="API Gateway objects and rollout state transitions" width="50%">
</p>

### Rollout API

A **rollout** is one execution of an agent on one input. It has a globally unique ID, an input, user-defined metadata, execution configuration, and a status:

- `QUEUING`: waiting for the Controller to start the agent;
- `RUNNING`: the agent is executing;
- `SUCCEEDED`: the execution completed successfully;
- `FAILED`: the execution ended with an error or timeout.

A rollout is not the same as a training example. Algorithms such as GRPO may create several independent rollouts from the same example so that the trainer can compare their rewards.

The trainer creates rollouts through the Rollout API. The Controller reads queued rollouts and updates their status as execution progresses. Each rollout can also contain append-only events, including:

- `model_request`, recorded automatically for each model call;
- `reward`, normally reported by the agent at the end of execution;
- custom events for diagnostics and monitoring.

Every event is associated with a specific rollout ID and is later exported as training data.

### OpenAI-compatible proxy

The Gateway also acts as a reverse proxy. The trainer registers one or more model inference endpoints, and the agent sends its model requests to a rollout-specific Gateway URL. The Gateway forwards each request to the registered model endpoint and records its prompt token IDs, response token IDs, and chosen-token log probabilities as a `model_request` event.

For example, an OpenAI Chat Completions request for a training rollout is sent to:

```text
POST /proxy/rollout/{rollout_id}/attempt/{attempt_id}/mode/train/openai/v1/chat/completions
```

The corresponding validation path uses `mode/val`. OpenAI-compatible clients can use the path through `/openai/v1` as their base URL and append `/chat/completions` normally.

Because the rollout ID is part of the proxy URL, every model call is automatically associated with the correct execution. An existing agent only needs to use the provided endpoint; it does not need to implement Agent Lightning's rollout or training logic.

## Rollout Controller

The Rollout Controller turns queued rollouts into real agent executions. It continuously reconciles rollout state in the API Gateway with the processes or Jobs it manages, and reports execution progress back to the Gateway.

<p align="center">
	<img src="../images/controller-reconciliation.jpg" alt="Controller reconciliation" width="75%">
</p>

The Controller supports two modes:

- **Local mode:** starts each rollout as a short-lived local subprocess. This mode is convenient for development and debugging when the agent and trainer dependencies can share one machine.
- **Kubernetes mode:** creates one Kubernetes Job for each rollout from a user-provided template. This mode isolates agent dependencies and supports concurrent execution on a self-hosted or on-premises cluster.

The API Gateway remains the source of truth for rollout status. If a process, Kubernetes watch, or network update is interrupted, the Controller retries reconciliation until the execution state converges.

## Customized Trainer

The Customized Trainer sits on top of `verl` and connects the training backend to the API Gateway. During each training step, it:

1. registers the current model inference endpoints;
2. creates one or more rollouts for each training input;
3. waits for enough rollouts to finish;
4. retrieves model requests, rewards, and other events;
5. converts the captured calls into `verl` training samples;
6. computes advantages and updates the policy.

The trainer also handles Agent Lightning-specific data processing. It merges consecutive model calls only when their token histories are exactly continuous, computes advantages at the rollout level, and supports rollout-level loss normalization.

## Handling failed rollouts

`SUCCEEDED` and `FAILED` are both terminal states. The Gateway rejects any transition out of them, and the trainer treats either state as "finished". The framework retries individual HTTP and model calls, but it does not retry a failed rollout, remove it from a training batch, or attach a failure-specific label. What to do with a failed rollout is your policy.

### What happens automatically

- The Gateway makes up to six attempts for each proxied model call on transport errors, timeouts, `408`, `409`, `429`, and `5xx`. Exhausted retries surface as an error to the agent.
- A rollout exceeding `agentlightning.rollout_timeout_seconds` is killed and marked `FAILED`: local runs record `local subprocess timed out`; Kubernetes runs record the Job failure reason.
- An uncaught exception in the agent's `run()` exits the worker non-zero and marks the rollout `FAILED`, with the exit code or Job reason in `error_message`.
- Returning `None` from `run()` is **not** a failure signal. The worker ignores the return value and exits with code 0, so the rollout becomes `SUCCEEDED` regardless of whether a reward event was recorded; a missing reward is then filled with `reward_fillna_value`.
- `FAILED` does not automatically exclude a rollout from training. Model calls recorded before the failure remain, and the trainer still builds a `CompletedRollout` from a `FAILED` rollout and passes it to the adapter alongside successful rollouts.
- If no `reward` event was recorded, `agentlightning.reward_fillna_value` (default `0.0`) is used as the reward. There is no check that the rollout succeeded.

The last point matters: an infrastructure failure after a few successful model calls can produce training rows labelled `0.0`, indistinguishable from a genuine zero-reward outcome.

### Decision table

"Framework" is behavior implemented in Agent Lightning v1.0. "Policy" is a decision you make in your agent or training driver.

| Situation | Framework behavior | Policy |
|---|---|---|
| Transient infrastructure failure, e.g. one model call times out or returns `5xx` | The Gateway retries that call with backoff. The rollout is unaffected while retries succeed. | Let the automatic retry work. Do not convert the error into a reward. If the call still fails, allow `FAILED`; re-enqueue with a **new** `rollout_id` only if you choose to retry the sample. |
| Persistent infrastructure failure, e.g. network down or a bad model endpoint | There is no rollout-level retry. Each affected rollout terminates on its own. | Stop the run and fix the network or endpoint. More rollout attempts will not repair an infrastructure fault and will waste the batch. |
| An exception in one sub-agent of a multi-agent rollout | Earlier sub-agents' `model_request` events are already stored. An uncaught exception yields `FAILED`, but the partial trace is still converted into training rows and a missing reward is filled with `reward_fillna_value`. | Decide explicitly: catch the exception and report a valid reward for the completed work, or discard the rollout at the training-data boundary. Do not assume partial traces are dropped for you. |
| A valid task failure versus an infrastructure failure | A `reward` event with a numeric `value` (negative values included) is used as-is. No `reward` event triggers `reward_fillna_value`. | Report task outcomes, including wrong answers, with an explicit `reward` event and finish normally so the rollout is `SUCCEEDED`. Never report `0.0` for an infrastructure error; let the rollout fail and filter it later. |
| Too many failed rollouts in one batch | The synchronous manager returns every terminal rollout; the asynchronous manager waits for every member of a group. There is no batch threshold or skip switch. | Check the reported `succeeded`/`failed` counters (or count failures in an `on_failed` hook) and abort the training step when your own threshold is exceeded. |

### Minimal example: separate task outcomes from infrastructure errors

The agent posts events to the `AGL_EVENT_URL` provided by the Controller. `solve()` and `expected` below are agent-specific.

```python
import json
import os
import urllib.request


def post_event(event_type: str, data: dict) -> None:
    request = urllib.request.Request(
        os.environ["AGL_EVENT_URL"],
        data=json.dumps({"event_type": event_type, "data": data}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('AGL_KEY', '')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5):
        pass


class InfrastructureError(RuntimeError):
    """Reserved for transport failures and timeouts, not for wrong answers."""


class MyAgent:
    def run(self) -> None:
        try:
            prediction = self.solve()
        except InfrastructureError as exc:
            post_event("infra_error", {"message": str(exc)})
            raise  # non-zero exit -> FAILED; deliberately no reward event

        # A wrong answer is a valid task outcome, reported with a real score.
        post_event("reward", {"value": 1.0 if prediction == self.expected else -1.0})
```

### Adapters and Triplet generation

Triplets come from the Gateway's `format="triplet"` view of `model_request` events; the rollout manager assembles them into `CompletedRollout`, and `RolloutAdapter` converts them into `verl` training rows. Model calls whose recorded status is `error` or whose HTTP status is `>= 400` are excluded from triplets.

Agent Lightning v1.0 has no `TracerTraceToTriplet` adapter, no `adapt()` hook, and no trainer setting for injecting a custom adapter. The built-in trainer constructs `RolloutAdapter` itself. A custom adapter that returns an empty triplet list is therefore not a supported configuration path on this version.

The supported customization seam is `agentlightning.hooks.RolloutHooks`, loaded from the `agentlightning.hooks` config path. Its `on_failed(rollout, store)` method is a lifecycle callback that runs when a rollout becomes `FAILED` and can add events through `store.add_event(...)`; it does not filter or remove the rollout. To exclude failed rollouts from training, filter the `completed_rollouts` list in your own training driver or trainer subclass before calling `RolloutAdapter.get_train_data_batch()`.

### Limitations

- There is no built-in per-rollout retry, batch-skip switch, or automatic suppression of zero-reward examples.
- `RolloutConfig` has no `max_attempts` or retry-condition fields. Snippets from older releases that call `emit_exception` or subclass `TracerTraceToTriplet` do not apply to v1.0.
- Re-enqueuing with the same `rollout_id` returns the existing rollout unchanged; use a fresh ID for a real retry.
- `reward_fillna_value` applies to every rollout without a recorded reward, not only to failed ones. Non-numeric `reward` values are ignored when computing the final reward.

## Configure the components

The following chapters describe the settings for each component. Start with the trainer to define how rollouts are created and converted into training samples, then configure the server and the Controller that execute them:

1. [Trainer Configuration](20-trainer-configuration.md)
2. [API Gateway Configuration](25-api-gateway-configuration.md)
3. [Controller Configuration](30-controller-configuration.md)
