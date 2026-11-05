# Sealed process credit

Process credit is an opt-in synchronous policy-gradient path for rewards attached to original model calls. It preserves the existing outcome-only path when `agentlightning.process_credit.enabled` is false. The initial supported training boundary is locked VERL 0.8.0 with its native FSDP/FSDP2 model engine. This is an experimental estimator/infrastructure feature, with no demonstrated learning or task-performance improvement.

## Objective and reward timing

For a complete rollout with actions indexed from zero, define instantaneous rewards `r_t` at their **information cutoff**. A process label declares both `target_event_id` (the action it judges) and `evidence_through_event_id` (the latest action whose information the checker used). Book its weighted value at the cutoff, including when the label retrospectively judges an earlier action. Delivery time and event sequence do not determine the cutoff. A terminal reward uses the last successful action, or null for a valid zero-action failure.

With fixed discount γ, positive normalization constant H, and K episodes scheduled before sampling:

\[
G_{i,t}=\sum_{u\geq t}\gamma^{u-t}r_{i,u},\qquad
J=\mathbb E\left[\frac{1}{H}\sum_t\gamma^t r_t\right],\qquad
c_{i,t}=\frac{\gamma^t(G_{i,t}-b_i)}{K H}.
\]

The registered `process_reinforce` loss sums `−c_{i,t} log π(token | context)` over **every generated token of each complete original action**, including sampled EOS. H defaults to 1024 and stays fixed; it is not the observed token length. K includes every scheduled episode, including zero-action failures, independent of the number of projected training rows. Tool, prompt, and padding positions receive zero credit. The loss multiplies by the FSDP data-parallel size to cancel averaged gradients and accumulates microbatches without a second normalization.

`baseline: zero` uses zero. `baseline: task_loo` uses the other independent episodes' complete discounted returns within the predeclared task group, with singleton fallback zero. It does not condition peers on surviving to a matching turn. Rollouts must be independent conditional on task and policy; shared stochastic state, adaptive group selection, or checks that inspect other episodes invalidate that assumption. A valid zero-action episode has empty discounted action return zero; its raw terminal outcome remains a separate diagnostic. The implementation uses that empty-sum return in LOO as well.

Under fixed sampling/termination, θ-independent environment dynamics and integrable rewards measurable at their declared cutoff, this is the usual unbiased **pre-clipping score-gradient estimator**. The cutoff is a checker attestation; the server cannot prove what information the checker inspected. Native gradient clipping and Adam updates do not inherit an unbiased-parameter-update guarantee. Matching serving version labels also does not prove numerical likelihood identity; the trainer logs captured/current logprob discrepancies without applying importance correction.

For example, `r=[.2,−.1,1]`, `γ=.5` gives `G=[.4,.4,1]` and outer-discounted values `[.4,.2,.25]`. With K=2,H=10 the per-action coefficients are `[.02,.01,.0125]`, repeated on each generated token. A label about action 0 using action 2 evidence contributes at action 2, not action 0.

## Agent integration

Use `ProcessCreditClient` or `AsyncProcessCreditClient` from `agentlightning.credit.client` with a native `httpx` client and explicit rollout ID, attempt ID, and policy version. The synchronous example is:

```python
from uuid import uuid4
from agentlightning.credit.client import ProcessCreditClient

credit = ProcessCreditClient(http_client, rollout_id, attempt_id, policy_version=step)
first = credit.complete(body, request_id="logical-call-0")
credit.emit_process_reward(
    .2, reward_id=uuid4().hex, source="local-prefix-check-v1",
    target_event_id=first.event_id, evidence_through_event_id=first.event_id,
)
credit.emit_terminal_reward(1, reward_id=uuid4().hex, source="local-final-check-v1")
receipt = credit.finalize()
```

Persist each request/reward ID before issuing it if replies may be lost. A retry uses the **same ID and body**. A genuinely new sample uses a new request ID even if its prompt is identical. The server assigns the original action event UUID and insertion sequence; duplicate waiters share a single forwarded operation. Disconnecting a waiter does not cancel capture. A request ID reused with a different body/index conflicts. Distinct concurrent logical calls within an attempt are rejected; finish or retry the outstanding call first.

A captured HTTP failure consumes a logical request ordinal but has no policy action. Successful original actions get contiguous credit indices, preserving any failed-call ordinal gaps in metadata. A possibly sampled response that is missing tokens/logprobs, has extra choices, or is lost through an upstream transport error makes the attempt incomplete. Such evidence cannot seal or silently become a zero reward. The process proxy does not automatically retry ambiguous upstream transport failures.

Each reward has a stable UUID, finite value and nonempty checker `source`; optional `reason` is retained. Identical reward retries are idempotent, conflicting replacements are rejected. Finalization supplies the exact ordered successful-action IDs, the exact reward-ID set and one terminal reward ID. It rejects in-flight calls, omitted actions/rewards, foreign or failed-action references, backwards cutoffs and inconsistent policy versions. Identical repeated finalization returns the same receipt. After sealing, identical retries are read-only; new evidence conflicts.

The native HTTP endpoints are:

- `POST /api/rollouts/{rollout_id}/attempt/{attempt_id}/credit/finalize`
- `GET /api/rollouts/{rollout_id}/attempt/{attempt_id}/credit`

Both require a marked process rollout and the existing server authentication. The manager selects the explicit terminal `last_attempt_id`, waits up to the configured monotonic finalization deadline and fails the collection if it cannot retrieve a complete seal. It never substitutes `reward_fillna_value` or trains a partial collection. Server state and logical request identities are in-memory and disappear on restart or rollout deletion.

## Native training configuration

Compose the existing native config as in [`examples/process_credit`](https://github.com/microsoft/agent-lightning/tree/main/examples/process_credit). The essential settings are:

```yaml
algorithm:
  adv_estimator: process_reinforce
  gamma: 0.5
  enable_rollout_level_advantage: false
  use_kl_in_reward: false
  rollout_correction: null
critic:
  enable: false
actor_rollout_ref:
  actor:
    strategy: fsdp
    policy_loss:
      loss_mode: process_reinforce
    ppo_epochs: 1
    entropy_coeff: 0
    use_kl_loss: false
    ulysses_sequence_parallel_size: 1
  rollout:
    temperature: 1
    top_p: 1
    top_k: -1
    'n': 2
agentlightning:
  async_rollout:
    enabled: false
  max_ppo_update_times: null
  process_credit:
    enabled: true
    gamma: 0.5
    baseline: task_loo
    normalization_tokens: 1024
    process_weight: 1
    terminal_weight: 1
    finalization_timeout_seconds: 300
```

`rollout.n` schedules independent episodes per task; each proxied model call requires `n=1`. Complete collection/group membership is fixed before enqueue makes episodes visible. One whole global collection becomes one optimizer minibatch and one epoch. Microbatching and DP padding preserve coefficients, K and every real row. Synthetic padding rows retain valid attention geometry with zero score mask, including ranks with no active generated tokens. An all-zero-action collection writes audit evidence and reports `process/actor_updates=0` without an optimizer call.

Process mode rejects asynchronous collection, critics, PPO objectives/replay, rollout correction/importance weights, KL or entropy terms, distillation, reward models, nonunit actor sequence/tensor/pipeline parallelism and remote HDFS checkpointing. The native model config must declare zero trainable dropout. Sampling must use temperature/top-p 1, disabled top-k, unit repetition penalty, zero presence/frequency penalty, complete nonstreamed single-choice responses, and no constrained decoding or stop-string stripping. The request allowlist admits model/messages/prompt context, fixed sampling fields, predeclared output-length limits, capture options and prompt-template context. Other extensions are rejected, including `min_tokens`, `response_format`, guided/structured decoding, hidden `extra_body`, logit processors and API tool/function constraints. Environment tool results may still be supplied as ordinary prompt context. The same validation runs on captured/replayed actions. These checks intentionally fail before workers/sample use rather than reinterpret unsupported settings. Serving/training weights, tokenizer and effective sampling distribution must match; quantized or custom kernels can violate that contract even when the version matches.

A complete `finish_reason=length` action is valid when generation was bounded before sampling and every sampled token/logprob is retained. A stopped action must retain its EOS token and logprob. Exceeding configured prompt/response/total projection limits aborts; no truncation or dropped-action fallback is available. Text trajectory merging preserves original action indices and token origins; a prefix mismatch starts a separate complete row. Transition mode preserves native image inputs/processor geometry. Image trajectory merging remains unsupported.

## Replay, limits and checkpoints

`agentlightning.credit.artifacts` provides `read_credit_artifact`, `write_credit_artifact` and `replay_credit` without importing torch/VERL or calling models. A versioned finite JSON artifact contains the sealed original events, configuration digest, reward attribution/cutoffs, deterministic returns, and (after projection) original token origins, masks, coefficients and scheduled group evidence. Replay recomputes rewards/returns and checks complete ordered token provenance, original context and fixed K/H/baseline math. Peer returns/hashes are collection evidence, not independently authenticated peer snapshots; auditing their truth requires the companion artifacts. Semantic hashes detect inconsistent modifications, not maliciously rewritten and rehashed evidence or checker authenticity.

The initial supported hard caps are 100000 credit events, 1024 actions, and 64 MiB of semantic evidence per attempt. Training config can tighten `max_events_per_rollout`, `max_actions_per_rollout` and `max_artifact_bytes`; larger values are explicitly unsupported. The native manager binds these limits into immutable enqueue metadata before sampling becomes visible, and the server constructs every attempt from that binding. Server operators may tighten `process_credit_limits` in server configuration; an enqueue request exceeding those caps fails before visibility/work. Reusing an existing rollout ID with a different process binding conflicts. Sealed snapshots include the admission limits, and replay rejects a configuration with different bounds. Event/action admission checks run before upstream forwarding; an unexpectedly large provider reply is rejected at capture, without claiming to bound unknown provider-side allocation or cost. Input reads are bounded before JSON parsing, reject duplicate keys/nonfinite constants, and writes atomically replace a finite JSON file. Float32 overflow and nonzero underflow are rejected for values projected to native training tensors. Raw finite labels may use float64 if their weighted results fit.

Before actor dispatch the trainer writes one artifact per snapshot under `process_credit.artifact_dir`, or `<trainer.default_local_dir>/process_credit` by default. If both output roots are absent, evidence stays in memory and `process/in_memory_evidence_only=1` makes that explicit. Source rollouts remain available until a successful update (or explicit all-zero skip). Checkpoints carry the algorithm/config digest and last completed collection hashes. Incompatible resumes fail before loading actor weights. This is offline evidence replay and completed-checkpoint resume; no exactly-once training guarantee spans crashes between actor update, artifact persistence and checkpoint publication. Never replay an artifact as a new on-policy sample after changing weights.

## Example and verification boundary

The arithmetic example makes three model calls, supplies deterministic local tool results, checks an immediate prefix and a retrospective label with a later cutoff, then seals an unweighted final outcome. It uses the existing local controller and native trainer. Ordinary validation remains outcome-only. With a configured Agent Lightning server/controller and the supported GPU training stack:

```bash
python -m examples.process_credit.train --model /path/to/zero-dropout-model --print-config
python -m examples.process_credit.train --model /path/to/zero-dropout-model
```

The printed configuration path is offline. The second command launches actual training and model generation; it was **not run** for this contribution.

CPU regressions cover exact enumerated stochastic-policy gradients (including variable horizon and zero-action failures), real HTTP proxy/client/seal→native manager→adapter→VERL `ppo_loss`, token lineage, retry/disconnect behavior, finite masked loss, and actual two-rank Gloo/DDP gradient aggregation with uneven microbatches and an all-masked rank. The native padding contract test binds the actual upstream pure-Torch `unpad_input` helper in tests only because the macOS CPU stack lacks FlashAttention; that optional test requires `einops`. Production GPU dispatch is unchanged. These checks do not execute CUDA/FSDP training or demonstrate learning gains. The disabled path has separate deterministic event/tensor/metric/RNG compatibility checks.
