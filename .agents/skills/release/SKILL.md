---
name: release
description: Prepare and publish stable Agent Lightning releases through the repository's version bump, pull-request checks, merge, tag, and PyPI trusted-publishing workflow. Use when asked to plan, cut, verify, or explain a release; treat nightly TestPyPI builds as a separate path.
---

# Release Agent Lightning

Merging a pull request does not publish a stable release. Stable publication is
triggered only by pushing a `v*` tag to the canonical repository; the tagged
commit is what gets tested, built, and uploaded.

## Establish the release state

1. Confirm the repository root, clean working tree, current branch, and remotes.
2. Resolve the canonical `OWNER/REPO` and its default branch with `gh repo view`.
   Identify the local remotes for that repository and the contributor fork by
   their URLs; do not assume particular remote names.
3. Inspect the release contract in:
   - `.github/workflows/pypi-release.yml`
   - `scripts/bump_version.sh`
   - `pyproject.toml`
   - `agentlightning/__init__.py`
4. Query the canonical repository's tags and compare them with the versions
   published at `https://pypi.org/pypi/agentlightning/json`. Confirm the target
   exists in neither place. If a tag is not represented on PyPI, or the proposed
   bump would skip an unpublished version, stop and obtain an explicit release
   decision. Never reuse or move a public release tag; PyPI versions are
   immutable.
5. Treat verified PyPI trusted-publisher configuration for the canonical
   repository and `pypi-release.yml` as a prerequisite. If it cannot be
   inspected directly, require confirmation from an authorized PyPI project
   owner before pushing the release tag.

## Prepare and merge the version pull request

Start a release branch from a freshly fetched canonical default branch, not
from another feature branch. The branch name is only a recommendation:

```bash
git fetch <canonical-remote> <default-branch>
git switch -c chore/release-vX.Y.Z <canonical-remote>/<default-branch>
scripts/bump_version.sh patch  # or minor / major
```

The bump script must keep these representations synchronized:

- `pyproject.toml`
- the Agent Lightning entry in `uv.lock`
- `agentlightning.__version__`

Review the version diff, but do not run the release tests or package build
locally. The pull request's GitHub checks are the verification gate.

```bash
git push -u <fork-remote> <release-branch>
gh pr create --repo OWNER/REPO \
  --base <default-branch> \
  --head <fork-owner>:<release-branch>
```

Commit the version change, push it to the fork, and open the pull request with
the GitHub CLI when those external actions are authorized. Use `gh pr checks
--watch` to follow the pull request through its required checks. If a check
fails, inspect it with `gh run view --log-failed`, correct the source on the same
branch, and resume watching. Once every required check has succeeded, merge the
pull request with `gh pr merge` using a merge method the repository permits.
Merging remains a distinct external action and requires authorization.

## Tag and publish the merged release

After the pull request merges, update the local default branch from the
canonical repository and use the resulting commit as the release candidate:

```bash
git switch <default-branch>
git pull --ff-only <canonical-remote> <default-branch>
```

Verify that the checked-out package version and runtime `__version__` both equal
`X.Y.Z`. Immediately query the canonical repository and PyPI again to ensure
that `vX.Y.Z` is still absent. Then create an annotated tag on the checked-out
merge commit and push it to the canonical repository:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push <canonical-remote> vX.Y.Z
```

The tag push starts the production PyPI publication, so obtain explicit
authorization immediately before it. Use `gh run list` and `gh run watch
--exit-status` to follow the `PyPI Release` workflow through its terminal result.
After success, verify that PyPI exposes the exact version and both the expected
wheel and source distribution, then report the workflow URL and PyPI outcome.

For a transient workflow failure, rerun only with authorization. For a source
or workflow defect, do not move the public tag; prepare a corrective release
version. A GitHub Release and release notes are optional, separate publication
actions and must not be created unless requested.

## Nightly distinction

`.github/workflows/pypi-nightly.yml` publishes timestamped `.dev` builds to
TestPyPI on its schedule or by manual dispatch. It does not create a stable
release and should not be substituted for the tag-driven process above.
