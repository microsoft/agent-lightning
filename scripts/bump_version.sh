#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.

# Bump the project version and keep pyproject.toml and uv.lock in sync.
set -euo pipefail

usage() {
    echo "Usage: $0 <major|minor|patch>"
}

if [ "$#" -ne 1 ]; then
    usage >&2
    exit 1
fi

BUMP="$1"
case "$BUMP" in
    major|minor|patch) ;;
    *)
        echo "ERROR: unsupported version bump: $BUMP" >&2
        usage >&2
        exit 1
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required to bump the project version." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_FILE="$PROJECT_ROOT/agentlightning/__init__.py"

if ! grep -q '^__version__ = ".*"$' "$VERSION_FILE"; then
    echo "ERROR: package version not found in $VERSION_FILE." >&2
    exit 1
fi

CURRENT_VERSION="$(uv version --project "$PROJECT_ROOT" --short)"
uv version --project "$PROJECT_ROOT" --bump "$BUMP" --no-sync
NEW_VERSION="$(uv version --project "$PROJECT_ROOT" --short)"

sed -i.bak "s/^__version__ = \".*\"$/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"
rm -f "$VERSION_FILE.bak"

echo "Bumped agentlightning version from $CURRENT_VERSION to $NEW_VERSION."
