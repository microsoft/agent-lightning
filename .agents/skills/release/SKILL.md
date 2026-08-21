---
name: release
description: Prepare and publish stable Agent Lightning releases through the repository's version bump, verification, tag, and PyPI trusted-publishing workflow. Use when asked to plan, cut, verify, or explain a release; treat nightly TestPyPI builds as a separate path.
---

# Release Agent Lightning

Merging a pull request does not publish a stable release. Stable publication is
triggered only by pushing a `v*` tag to `microsoft/agent-lightning`; the tagged
commit is what gets tested, built, and uploaded.

## Establish the release state

1. Confirm the repository root, clean working tree, current branch, and remotes.
   The canonical repository must be the `upstream` remote and the contributor
   fork should be `origin`.
2. Fetch `upstream/main` and inspect the real release contract in:
   - `.github/workflows/pypi-release.yml`
   - `scripts/bump_version.sh`
   - `pyproject.toml`
   - `agentlightning/__init__.py`
3. Query upstream tags directly and compare them with the versions actually
   published at `https://pypi.org/pypi/agentlightning/json`. Confirm the target
   exists in neither place. If a tag is not represented on PyPI, or the proposed
   bump would skip an unpublished version, stop and obtain an explicit release
   decision. Never reuse or move a public release tag; PyPI versions are
   immutable.
4. Treat verified PyPI trusted-publisher configuration for
   `microsoft/agent-lightning` and `pypi-release.yml` as a prerequisite. If it
   cannot be inspected directly, require confirmation from an authorized PyPI
   project owner before pushing the release tag.

## Prepare the version

Start a release branch from freshly fetched `upstream/main`, not from another
feature branch. The branch name below is a recommendation, not a repository
requirement:

```bash
git fetch upstream main
git switch -c chore/release-vX.Y.Z upstream/main
scripts/bump_version.sh patch  # or minor / major
```

The bump script must keep these three representations synchronized:

- `pyproject.toml`
- the Agent Lightning entry in `uv.lock`
- `agentlightning.__version__`

Review the diff and reject unrelated changes. Run the commands owned by the
release workflow before proposing the version bump:

```bash
uv sync --frozen --no-default-groups --extra dev --group dev
uv run --locked --no-sync pytest -v --durations=20 \
  tests/server \
  tests/controller \
  tests/test_package.py \
  tests/examples/test_swe_smith_images.py
ARTIFACT_DIR=$(mktemp -d)
uv build --no-sources --out-dir "$ARTIFACT_DIR"
python -m tarfile -l "$ARTIFACT_DIR"/*.tar.gz
python -m zipfile -l "$ARTIFACT_DIR"/*.whl
```

Commit, push, and open the version-bump pull request only when each action is
authorized. Merging is another external action: merge only after the required
checks pass and the user explicitly authorizes it.

## Publish the release

After the version-bump pull request merges:

1. Fetch `upstream/main` and identify the exact merge commit intended for the
   release. Verify that its package version and runtime `__version__` both equal
   `X.Y.Z`.
2. Query the remote immediately before release and verify again that `vX.Y.Z`
   is absent locally, on `upstream`, and on PyPI. Fetching only `upstream/main`
   is not a remote-tag check.
3. Immediately before pushing, explain that the tag push starts the production
   PyPI publication and obtain explicit authorization for that push.
4. Create an annotated tag on the exact merge commit and push it to `upstream`:

```bash
git tag -a vX.Y.Z <release-merge-sha> -m "vX.Y.Z"
git push upstream vX.Y.Z
```

The `PyPI Release` workflow then verifies the tag and runtime versions, runs the
core tests, builds the source distribution and wheel, lists their contents, and
publishes through OIDC trusted publishing. Watch that workflow through its
terminal result. After success, verify that PyPI exposes the exact version and
both the expected wheel and source distribution, then report the run URL and
PyPI outcome.

For a transient failure, rerun only with authorization. For a source or workflow
defect, do not move the public tag; prepare a corrective release version. A
GitHub Release and release notes are optional, separate publication actions and
must not be created unless requested.

## Nightly distinction

`.github/workflows/pypi-nightly.yml` publishes timestamped `.dev` builds to
TestPyPI on its schedule or by manual dispatch. It does not create a stable
release and should not be substituted for the tag-driven process above.
