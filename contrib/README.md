# Contrib Area

Contributions that do not naturally live under the main `agentlightning/`, `examples/`, or `docs/` trees land here.
Use this space for experimental integrations, community-maintained add-ons, and curated recipes that help users assemble Agent-lightning components quickly.

## Directory map

- `agentlightning/` — Python namespace packages, utilities, and adapters that extend the core `agentlightning` distribution without bloating the main runtime. The inner `agentlightning/contrib/` package matches the layout of the published wheel so downstream users can `import agentlightning.contrib.<feature>`.
- `recipes/` — Task-focused example bundles that highlight how to wire up Agent-lightning core or contrib components. Each recipe should include a README file and be self-contained.
- `scripts/` — Shared scripts, data download instructions, and automation snippets that keep the contrib area working.
When adding new folders, document the intent in a local README, update `CODEOWNERS`, and link to companion docs or
examples so maintainers can trace ownership quickly.

Questions or proposals for new subtrees can be raised in GitHub issues or discussions before opening a PR. For detailed checklists covering scope decisions, documentation requirements, testing, and CI expectations, see the "Agent-lightning Contrib" section of [`docs/community/contributing.md`](../docs/community/contributing.md). This keeps the contrib area focused on high-signal additions that demonstrate how to extend Agent Lightning responsibly.
