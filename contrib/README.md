# Contrib Area

This tree hosts experimental integrations, third-party recipes, and curated recipes that are not ready for the main `agentlightning/`, `examples/`, or `docs/` trees. Treat it as an incubator: keep contributions self-contained, clearly owned, and reproducible so downstream users can vendor them without guesswork.

## When to add something here

- You are iterating on a runtime extension that would bloat the primary `agentlightning/` namespace.
- You want to share a recipe that assembles existing components for a focused agent training or optimization workflow and needs more context than the main examples directory allows.
- You need automation scripts or download helpers that will help the community but should not live under `scripts/` at the repo root.

If a contribution starts depending on core release cadence, tight CI guarantees, or repo-wide infrastructure, talk to maintainers about graduating it out of `contrib/`.

## Directory map

- `agentlightning/` — Namespace packages, utilities, and adapters that extend the published wheel. Place new code under `agentlightning/contrib/<feature>/` so `import agentlightning.contrib.<feature>` works for downstream users.
- `recipes/` — Task-focused example bundles that solve a specific problem and derive certain results. Each recipe belongs in its own directory with a README that documents usage, result reports, and ownership.
- `scripts/` — Shared automation, dataset download steps, or reproducibility helpers that support the contrib modules above.

When adding folders, document the intent in a local README, link to companion docs or examples, and update `CODEOWNERS` so future fixes reach the right reviewers quickly.

Questions or proposals for new subtrees can be discussed in Discord, GitHub issues, or GitHub Discussions before opening a PR. For the canonical requirements and review checklist, see the “Agent-lightning Contrib” section of [`docs/community/contributing.md`](../docs/community/contributing.md).
