# Maintainer Guide

This guide describes the day-to-day responsibilities for Agent Lightning maintainers—how to bump versions, run releases, and keep the minimal CI and documentation automation healthy.

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
    git pull origin main
    git checkout -b stable/v2.0.x  # adjust to the new series
    git push origin stable/v2.0.x
    ```

    All future changes to the stable branch must land via pull requests.

### Preparing the Next Release

When it is time to publish the next version:

1. **Draft release notes** in `docs/changelog.md`, collecting every notable change since the previous tag.
2. **Open a release PR** targeting `main` (for minor/major) or the relevant stable branch (for patch releases). Use the title `[Release] vX.Y.Z`.
3. **Run validation** through the pull request's `tests.yml` checks and the relevant local example smoke tests. Investigate and resolve any failures.
4. **Merge the release PR** once notes are final and CI is green.
5. **Tag the release** from the branch you just merged into:

    ```bash
    git checkout main          # minor/major releases
    git checkout stable/vX.Y.Z # patch releases

    git pull
    git tag vX.Y.Z -m "Release vX.Y.Z"
    git push origin vX.Y.Z
    ```

    Pushing the tag deploys the versioned documentation.

6. **Publish the GitHub release** using the drafted notes and confirm the documentation site reflects the new version. This repository does not publish a separate PyPI distribution; users install fork releases from Git tags.

## Repository Automation

The repository intentionally keeps only two GitHub Actions workflows:

- `tests.yml` runs CPU tests, linting, documentation checks, and dashboard checks for pull requests and pushes to maintained branches. It can also be started manually.
- `docs.yml` publishes versioned documentation from `main` and release tags.

GPU, cloud-provider, benchmark, and example end-to-end tests are run locally or in purpose-built infrastructure when needed. Their smoke-test commands remain documented in the corresponding example README files.

Backports to stable branches are performed manually with `git cherry-pick`, followed by the same pull request checks used for `main`.

## Repository Configuration

Configure the following GitHub repository settings:

- Enable GitHub Pages with GitHub Actions write access for versioned documentation.
- Protect `main` and stable branches, require the relevant status checks, and enable private vulnerability reporting.

Forks do not inherit branch protection, Pages settings, or repository labels from another repository. Configure those in GitHub before relying on the corresponding automation.
