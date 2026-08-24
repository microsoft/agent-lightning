---
name: release
description: Prepare and publish stable Agent Lightning releases through the repository's version bump, pull-request checks, merge, tag, PyPI trusted-publishing, and versioned-documentation workflows. Use when asked to plan, cut, verify, or explain a release; treat nightly TestPyPI builds as a separate path.
---

# Release Agent Lightning

Merging a pull request does not publish a stable release. Stable publication is
triggered only by pushing a `v*` tag to the canonical repository; the tagged
commit is what gets tested, built, and uploaded. That same tag push also deploys
versioned documentation and moves the public `stable` alias, so a release has two
public side effects, not one.

## Establish the release state

1. Confirm the repository root, clean working tree, current branch, and remotes.
2. Resolve the canonical `OWNER/REPO`, its default branch, and its permitted
   merge methods with
   `gh repo view OWNER/REPO --json nameWithOwner,defaultBranchRef,mergeCommitAllowed,rebaseMergeAllowed,squashMergeAllowed`.
   Identify the local remotes for that repository and the contributor fork by
   their URLs; do not assume particular remote names or merge settings.
3. Inspect the release contract in:
   - `.github/workflows/pypi-release.yml`
   - `.github/workflows/docs.yml`
   - `.github/workflows/tests.yml`
   - `scripts/bump_version.sh`
   - `pyproject.toml`
   - `agentlightning/__init__.py`
4. Confirm the canonical default branch is already green before branching from
   it. Resolve its current commit with
   `gh api repos/OWNER/REPO/commits/<default-branch>` and inspect that commit's
   check runs; a general recent-run listing can omit or mix commits. A release
   branch inherits every failure that main is carrying.
5. Query the canonical repository's tags and compare them with the versions
   published at `https://pypi.org/pypi/agentlightning/json`. Confirm the target
   version exists in neither place, and stop for an explicit release decision
   when either of these holds:
   - A tag exists with no matching PyPI version. A published version is
     immutable, and its tag must never be reused or moved. A tag that never
     published is a different situation and still needs a human decision,
     informed by why it did not publish. See "Recovering a tag that never
     published" below.
   - The proposed bump would skip a version that was tagged but never published.
6. Treat verified PyPI trusted-publisher configuration for the canonical
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

The script updates the project version with uv and then edits
`agentlightning/__init__.py` separately. If it fails or is interrupted between
those writes, only some of the three version files may be updated. Inspect
`git diff` after any failure and restore or reconcile all three files before
retrying; blindly rerunning a partial patch bump can advance the version twice.

The bump rewrites exactly three files. Confirm that with `git diff --stat`:

- `pyproject.toml`
- the `agentlightning` entry in `uv.lock`
- `agentlightning.__version__` in `agentlightning/__init__.py`

Other version strings in the tree, such as the FastAPI `version` in
`agentlightning/server/app.py`, are deliberately outside the bump. Leave them
alone; changing them is a separate pull request, not release work.

Review the version diff, but do not run the release tests or package build
locally as a matter of course. `tests.yml` runs a broader test suite and the
same package build on the pull request, covering the narrower tests and build
that `pypi-release.yml` will run on the tag. The pull request's GitHub checks
are therefore the verification gate. Reproduce a single failure locally only
when the workflow logs are not enough to fix it.

Commit the version change, push it to the fork, and open the pull request with
the GitHub CLI when those external actions are authorized:

```bash
git commit -am "Bump version to X.Y.Z"
git push -u <fork-remote> <release-branch>
gh pr create --repo OWNER/REPO \
  --base <default-branch> \
  --head <fork-owner>:<release-branch> \
  --title "Bump version to X.Y.Z" \
  --body "Prepare the vX.Y.Z release."
```

`gh pr create` refuses to run without `--title` and `--body` outside an
interactive terminal, and every `gh` call needs `--repo OWNER/REPO` so it acts
on the canonical repository rather than the fork.

Follow the pull request through its required checks with
`gh pr checks <pr> --repo OWNER/REPO --watch`. If a check fails, take the run id
from that output, inspect it with
`gh run view <run-id> --repo OWNER/REPO --log-failed`, correct the source on the
same branch, and resume watching. Once every required check has succeeded,
extract the reviewed head and pass both a permitted merge-method flag from step
2 and `--match-head-commit` to `gh pr merge`:

```bash
HEAD_SHA="$(gh pr view <pr> --repo OWNER/REPO --json headRefOid --jq .headRefOid)"
gh pr merge <pr> --repo OWNER/REPO <merge-method-flag> \
  --match-head-commit "$HEAD_SHA"
```

Replace `<merge-method-flag>` with one permitted flag discovered in step 2:
`--merge`, `--rebase`, or `--squash`.

Committing, pushing, opening the pull request, and merging are each distinct
external actions and each requires authorization.

## Tag and publish the merged release

After the pull request merges, update the local default branch from the
canonical repository, then confirm that the commit you are about to tag is the
one this pull request produced and not a later commit that landed behind it:

```bash
git switch <default-branch>
git pull --ff-only <canonical-remote> <default-branch>
gh pr view <pr> --repo OWNER/REPO --json mergeCommit
git rev-parse HEAD
```

If HEAD has moved past the merge commit, tag the merge commit explicitly instead
of HEAD.

`pypi-release.yml` fails the release when the packaged version does not equal the
tag without its leading `v`, or when it does not equal the runtime
`__version__`. Check both before tagging:

```bash
uv version --short
grep '^__version__' agentlightning/__init__.py
```

The workflow itself reads the runtime value as
`python -c 'from agentlightning import __version__; print(__version__)'` from
the repository root before its dependency-sync step, so Python resolves the
checkout through the current working directory. Read the file directly for the
local pre-tag check; `agentlightning/__init__.py` assigns `__version__` as a
single literal, making that check independent of the active Python environment.

GitHub reads workflow files as they exist **at the tagged commit**, not at the
tip of the default branch. Confirm that the commit being tagged actually
contains `.github/workflows/pypi-release.yml` with its `v*` trigger; a commit
that predates the workflow will never publish, however the tag is pushed.

Immediately query the canonical repository and PyPI again to ensure that
`vX.Y.Z` is still absent. Then create an annotated tag on the release commit and
push it to the canonical repository:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z" <release-commit>
git push <canonical-remote> vX.Y.Z
```

The tag push starts the production PyPI publication, so obtain explicit
authorization immediately before it.

## Follow both tag-triggered workflows

One tag push starts two workflows, and both belong to the release:

- `PyPI Release` (`pypi-release.yml`) re-checks the version against the tag,
  runs the tests, builds the wheel and source distribution, and uploads them to
  PyPI through trusted publishing.
- `Deploy Documentation` (`docs.yml`) runs
  `mike deploy --push --update-aliases X.Y.Z stable`, which publishes the
  versioned documentation and repoints the public `stable` alias at this
  release.

Look up each run by workflow and tag rather than selecting from an unfiltered
recent-run list:

```bash
gh run list --repo OWNER/REPO --workflow pypi-release.yml \
  --branch vX.Y.Z --event push --limit 1
gh run list --repo OWNER/REPO --workflow docs.yml \
  --branch vX.Y.Z --event push --limit 1
```

Confirm both runs have the expected tag commit, then follow them to a terminal
result with `gh run watch <run-id> --repo OWNER/REPO --exit-status`. After
`PyPI Release` succeeds, verify that PyPI exposes the exact version with both
the expected wheel and source distribution. After `Deploy Documentation`
succeeds, verify that the published site serves `X.Y.Z` and that `stable`
serves that same release. Do not require an HTTP redirect: Mike aliases can
serve directly from the alias path with a `200` response. Fetch the versioned
and `stable` entry pages and compare their content; byte-for-byte equality is
the clearest proof for a static deployment. If path-dependent markup prevents
an exact match, verify a release-specific marker such as the version selector
or canonical metadata instead. A successful response from `stable` alone is
not proof that the alias moved. A green PyPI job with a failed documentation
job is a half-finished release: report both workflow URLs and both outcomes.

For a transient workflow failure, rerun only with authorization. For a source
or workflow defect, do not move the public tag; prepare a corrective release
version. A GitHub Release and release notes are optional, separate publication
actions and must not be created unless requested.

## Recovering a tag that never published

Separate the mechanics from the policy before proposing a recovery.

The mechanics: pushing a tag that already exists and points at the same commit
changes no ref, so it starts no workflow run. Creating a tag, moving one to a
different commit, or deleting and recreating one does change the ref and does
start a run. What that run executes is the workflow file at the tagged commit,
so a tag on a commit from before `pypi-release.yml` existed starts no PyPI
publication no matter how it is pushed. Run
`git ls-tree --name-only <tag> .github/workflows/` before assuming a re-push
would help.

The policy: never move or reuse a tag whose version is on PyPI. That version is
immutable, so a re-run could only fail at upload, and consumers who already
resolved the tag would silently get different code.

Between those, a tag that never published is a decision for a release owner,
not a default action. Releasing the next version from a commit that carries the
current workflow is usually simpler and always safer than resurrecting the old
tag. Note that a non-publishing tag may still have had effects: `docs.yml` has
carried the `v*` trigger for longer than `pypi-release.yml`, so an older tag can
have deployed documentation and moved `stable` without ever reaching PyPI.

## Nightly distinction

`.github/workflows/pypi-nightly.yml` publishes timestamped `.dev` builds to
TestPyPI on its schedule or by manual dispatch. It does not create a stable
release and should not be substituted for the tag-driven process above.
