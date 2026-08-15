# SHAPER Runtime Extension

SHAPER evolves two model-external resources around a frozen Agent Lightning
agent: a textual skill and executable context-construction code. It diagnoses
observable rollout transitions, summarizes episode failures into textual
gradients, and performs a sequential skill-then-harness beam search.

The runtime lives in `contrib` because Agent Lightning's core resource union
does not yet have a code-harness resource. SHAPER transports both artifacts as
`PromptTemplate` values. Integrations read harness `template` text as source;
they must never call `PromptTemplate.format()` on that source.

Install the extension beside the matching Agent Lightning release:

```bash
python -m pip install -e .
python -m pip install -e contrib/agentlightning/contrib/shaper
```

Install the matching core checkout together with the extension when its exact
core version has not yet been published.

Public API:

- `SHAPER`: the two-stage optimization algorithm.
- `SHAPERTraceAdapter`: extracts structured round and episode records.
- `RoundRecord` and `EpisodeMetadata`: the agent-to-algorithm trace contract.
- `PythonHarnessValidator`: static and isolated-process harness validation.
- `emit_round_record` and `emit_episode_metadata`: rollout instrumentation.

Generated harnesses are never executed in the Trainer or simulator process.
Validation and every runtime call use the same restricted isolated interpreter
with finite CPU, memory, output, and wall-time limits. This is fault containment,
not an OS sandbox; use a container or VM for code from an untrusted author.

See [`contrib/recipes/shaper/README.md`](../../../recipes/shaper/README.md)
for VLABench/ESI-Bench environment setup, training, and evaluation commands.
