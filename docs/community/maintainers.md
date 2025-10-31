# Maintainer Guide

This guide documents the day-to-day workflows for the Agent Lightning maintainers, including how to cut a release, interact with CI, and backport fixes.

## Release Workflow

Follow this checklist throughout a release cycle.

### After the Last Release

We start from the time when the previous release has just been out.

Agent-lightning follows a **bump version first** strategy. We bump to next version immediately after a release is out. To bump the version, update the following files:

- `pyproject.toml`: Update the `version` field.
- `agentlightning/__init__.py`: Update the `__version__` variable if present.
- `uv.lock`: Run `uv lock` to update the lockfile with the new version.

We also bump dependency versions as needed to keep up with ecosystem changes.

```bash
uv lock --upgrade
```

If the last release is a new major or minor version, create a new stable branch from `main`:

```bash
git checkout main
git pull upstream main
git checkout -b stable/v2.0.x  # <-- adjust version as needed
git push upstream stable/v2.0.x
```

Notice that the stable branch requires merging via pull requests once you have pushed it to the upstream remote.

### Preparing for a New Release

When it's time to prepare a new release, follow these steps below.

1. **Prepare Release Note Draft**: Start collecting all the changes since the last release, and write the changes in `docs/changelog.md` under a new heading for the upcoming version.
2. **Open a Pull Request**: Open a PR against `main` (if it's a minor/major release) or the relevant stable branch (if it's a patch release) with the draft release notes. Use the title `[Release] vX.Y.Z`.
3. **Run CI checks**: Label the PR with `ci-all` and comment `/ci` to run all the tests, including GPU and example pipelines. Address any issues that arise.
4. **Merge the release PR**: Once all checks pass and the release notes are finalized, merge the PR.
5. **Create a tag for the release**: After merging, create a tag that matches the version:

    ```bash
    git checkout main  # <-- if it's a minor/major release
    git checkout stable/vX.Y.Z  # <-- if it's a patch release

    git pull
    git tag vX.Y.Z -m "Release vX.Y.Z"
    git push upstream vX.Y.Z
    ```

    Pushing the tag triggers the PyPI publish and documentation deployment workflows.

6. **Create the GitHub release**: Use the prepared notes from the PR to create a new GitHub release. Verify that the docs site shows the new version and the new package are available on PyPI.

## Working with CI Labels and `/ci`

Long-running jobs such as GPU training or example end-to-end runs are opt-in on pull requests. To trigger them:

1. Add one or more of the following labels to the PR before issuing the command:
    - `ci-all` — run every repository-dispatch aware workflow.
    - `ci-gpu` — run the GPU integration tests (`tests-full.yml`).
    - `ci-apo`, `ci-calc-x`, `ci-spider`, `ci-unsloth`, `ci-compat` — run the corresponding example pipelines.
2. Comment `/ci` on the pull request. The `issue-comment` workflow will acknowledge the command and track results inline.
3. Remove labels once the signal is collected to avoid accidental re-triggers.

Use `/ci` whenever a change affects shared infrastructure, dependencies, or training logic that requires extra validation beyond the default PR checks.

!!! note

    When invoking `/ci` on PR, the workflow always runs against the workflow defined on the main branch. It then checks out the PR changes within the workflow. So if you need to modify the workflow itself, push the changes to a branch on the first-party repository first, and then run:

    ```bash
    gh workflow run examples-xxx.yml --ref your-branch-name
    ```

## Backporting Pull Requests

We rely on automated backports for supported stable branches.

1. Decide which stable branch should receive the fix (for example, `stable/v0.2.x`).
2. Before merging the PR into `main`, add a label matching `stable/<series>` (e.g., `stable/v0.2.x`).
3. The `backport.yml` workflow creates a new PR named `backport/<original-number>/<target-branch>` authored by `agent-lightning-bot`.
4. Review the generated backport PR, ensure CI passes, and merge it into the target branch.
5. If conflicts arise, push manual fixes directly to the backport branch and re-run `/ci` as needed.

Keep the stable branches healthy by cherry-picking only critical fixes and ensuring their documentation and example metadata stay in sync with the release lines.
