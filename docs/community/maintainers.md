# Maintainer Guide

This guide describes the day-to-day responsibilities for Agent Lightning maintainers—how to bump versions, run release ceremonies, interact with CI, and backport fixes safely.

## Release Workflow

Follow this checklist throughout each release cycle.

### Immediately After Shipping

Agent Lightning uses a **bump-first** strategy. As soon as a release is published:

1. Update version metadata:
    - `pyproject.toml`: bump the `version`.
    - `agentlightning/__init__.py`: update `__version__` if it exists.
    - `uv.lock`: refresh the lock file after the bump.
2. Refresh dependency pins as needed:
    ```bash
    uv lock --upgrade
    ```

3. For a new minor or major release, create a stable branch from `main`:
    ```bash
    git checkout main
    git pull upstream main
    git checkout -b stable/v2.0.x  # adjust to the new series
    git push upstream stable/v2.0.x
    ```

    All future changes to the stable branch must land via pull requests.

### Preparing the Next Release

When it is time to publish the next version:

1. **Draft release notes** in `docs/changelog.md`, collecting every notable change since the previous tag.
2. **Open a release PR** targeting `main` (for minor/major) or the relevant stable branch (for patch releases). Use the title `[Release] vX.Y.Z`.
3. **Run extended CI** by labeling the PR with `ci-all` and commenting `/ci`. Investigate and resolve any failures.
4. **Merge the release PR** once notes are final and CI is green.
5. **Tag the release** from the branch you just merged into:

    ```bash
    git checkout main          # minor/major releases
    git checkout stable/vX.Y.Z # patch releases

    git pull
    git tag vX.Y.Z -m "Release vX.Y.Z"
    git push upstream vX.Y.Z
    ```

    Pushing the tag publishes to PyPI and deploys the documentation.

6. **Publish the GitHub release** using the drafted notes, and confirm the docs site and PyPI listing reflect the new version.

## Working with CI Labels and `/ci`

GPU suites and example end-to-end runs are opt-in. To trigger them on a pull request:

1. Apply the appropriate labels before issuing the command:
    - `ci-all` for every repository-dispatch workflow.
    - `ci-gpu` for GPU integration tests (`tests-full.yml`).
    - `ci-apo`, `ci-calc-x`, `ci-spider`, `ci-unsloth`, `ci-compat` for the individual example pipelines.
2. Comment `/ci` on the PR. The `issue-comment` workflow acknowledges the request and tracks job results inline.
3. Remove the labels once you have the signal to avoid accidental re-runs.

Use `/ci` whenever changes touch shared infrastructure, dependencies, or training loops that require coverage beyond the default PR checks.

!!! note

    `/ci` always executes the workflow definitions on the current `main` branch, then checks out the PR diff. If you need to test workflow modifications, push the changes to a branch in the upstream repo and run:

    ```bash
    gh workflow run examples-xxx.yml --ref your-branch-name
    ```

## Backporting Pull Requests

Supported stable branches rely on automated backports:

1. Identify the target branch (for example `stable/v0.2.x`).
2. Before merging the original PR into `main`, add the matching `stable/<series>` label (e.g. `stable/v0.2.x`).
3. The `backport.yml` workflow opens a follow-up PR named `backport/<original-number>/<target-branch>` authored by `agent-lightning-bot`.
4. Review the generated PR, ensure CI is green, and merge into the stable branch.
5. Resolve conflicts by pushing manual fixes to the backport branch and re-running `/ci` if required.

Keep stable branches healthy by cherry-picking only critical fixes and ensuring documentation and example metadata stay aligned with each release line.
