---
name: agent-lightning
description: >-
  Provides the action space, tradeoffs, and evaluation context for improving an
  editable AI agent against a benchmark while preserving its deployment contract.
  Use when optimizing agent accuracy, cost, latency, or reliability.
---

# Agent Lightning

Agent optimization is a search over interacting choices. The useful question is
not which architecture is most sophisticated, but which change moves the requested
accuracy, cost, latency, and reliability frontier for this agent.

The optimizer's development budget and the resulting agent's per-run cost are
different quantities. More development budget creates room to learn; it does not
imply that the deployed agent should spend more on every task.

Your remaining development budget is reported at `/artifacts/cost_budget.json`
(`{spent, budget, remaining}`), refreshed as you work. Read that file to see how
much is left, and keep iterating — measure, edit, re-score — while meaningful
budget remains; do not stop at the first plausible result. A run is finished not
because one change worked, but because further measured changes no longer improve
the frontier within the budget you still have. If `remaining` is large, there is
more search to do: ground more cases, probe a lever you have not tested, or add
reps to resolve a noisy comparison. Check `remaining` again after each expensive
step so the decision to stop is evidence-based, not a default.

## Action space

| Lever                    | What it changes                                          | Useful signal                                                   | Main tradeoff                              |
| ------------------------ | -------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| Input grounding          | Information and state visible to the model               | Relevant deployment-visible context is missing                  | Longer context can distract or cost more   |
| Output contract          | Representation, types, schema, files, and terminal state | Work looks reasonable but is rejected or unreadable             | Can overfit evaluator quirks               |
| Prompt                   | Interpretation, priorities, and constraints              | Instructions are misunderstood or important details are ignored | Prompt gains can be brittle                |
| Tools                    | Deterministic inspection, computation, and execution     | The model is approximating work a tool can do reliably          | More code and new failure modes            |
| Model                    | Base capability and knowledge                            | The primary cannot solve grounded cases                         | Cost, latency, and availability            |
| Reasoning effort         | Computation used by the primary call                     | Grounded hard cases remain                                      | Cost and latency can grow nonlinearly      |
| Failure isolation        | Whether one failure damages other work                   | Individual tasks crash, time out, or corrupt shared state       | Isolation does not recover the failed task |
| Conditional repair       | A second attempt informed by failure evidence            | Deployment-visible checks expose a recoverable failure          | Extra calls and possible regressions       |
| Routing                  | Different handling for different task classes            | Difficulty or failure risk varies predictably                   | Router mistakes and operational complexity |
| Planning and interaction | State, ordering, and tool use across multiple steps      | Long tasks lose goals or ignore observations                    | More state and control-flow overhead       |
| Retrieval                | Facts supplied from an available corpus                  | Correct answers depend on external knowledge                    | Retrieval errors and added latency         |
| Critique or selection    | Additional views or candidates                           | Independent attempts expose different useful information        | Multiplied calls, cost, and latency        |

## Reading the evidence

Different failures expose different amounts of information:

- Exceptions, missing artifacts, invalid schemas, and timeouts are objective
  signals. They can support deterministic checks or focused recovery.
- A valid-looking but wrong answer may expose no label-free repair signal. More
  calls with the same information can repeat the same mistake.
- Repeated failures across different attempts suggest shared blindness, a contract
  mismatch, or missing capability. Diverse failures make routing, critique, or
  selection more plausible.
- A development gain that disappears under validation may come from randomness,
  memorized examples, training-only fields, or a different deployment path.
- An unavailable measurement is unknown evidence, not proof that a candidate
  improved or regressed.

Levers interact. More reasoning cannot recover information the model never sees.
Repair cannot fix a systematic contract error when the retry receives no new
evidence. Failure isolation preserves the batch but does not repair an item. A
global model or effort increase and conditional escalation occupy different points
on the frontier.

## Evaluation context

Agent evaluations are often stochastic. A score can improve while many individual
cases regress, and a single strong result can be a lucky draw. Fixed cases,
validation splits, repeated runs, frozen primary outputs, and small probes are
different ways to reduce uncertainty; their value depends on the decision and the
available budget.

Comparisons are easiest to interpret when the checkpoint, cases, model, effort,
concurrency, scorer, and execution path are held constant except for the variable
being studied. Accuracy is only one result: completion rate, downside, cost, latency,
and variance can change the decision.

Development may expose labels, metadata, or tools that do not exist at deployment.
An improvement that depends on them is not a deployed improvement. Likewise,
measurement plumbing can fail independently of the agent being tested.

## Boundaries

- Preserve the target's external interface and deployment environment.
- Do not expose held-out labels or training-only answer fields to the deployed path.
- Treat the scorer and evaluation contract as immutable measurement surfaces.
- Leave a coherent measured checkpoint, not an unfinished or partially tested edit.
