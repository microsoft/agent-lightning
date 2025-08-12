#!/bin/bash

# Script to bump version in pyproject.toml
# Usage: ./bump_version.sh <new_version>

set -e

# Check if version argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No version specified"
    echo "Usage: $0 <new_version>"
    echo "Examples: $0 1.2.3, $0 1.2.3a1, $0 1.2.3b2, $0 1.2.3rc1, $0 1.2.3.post1, $0 1.2.3.dev1"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (Python PEP 440 compliant)
if ! echo "$NEW_VERSION" | grep -E '^[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?$' > /dev/null; then
    echo "Error: Invalid version format"
    echo "Version should follow PEP 440 (e.g., 1.2.3, 1.2.3a1, 1.2.3b2, 1.2.3rc1, 1.2.3.post1, 1.2.3.dev1)"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"

# Check if pyproject.toml exists
if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "Error: pyproject.toml not found at $PYPROJECT_FILE"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep '^version = ' "$PYPROJECT_FILE" | sed 's/version = "\(.*\)"/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    echo "Error: Could not find current version in pyproject.toml"
    exit 1
fi

echo "Current version: $CURRENT_VERSION"
echo "New version: $NEW_VERSION"

# Update version in pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT_FILE"

# Remove backup file
rm -f "$PYPROJECT_FILE.bak"

echo "Successfully bumped version from $CURRENT_VERSION to $NEW_VERSION"

# Update __init__.py if it exists with version
INIT_FILE="$PROJECT_ROOT/agentlightning/__init__.py"
if [ -f "$INIT_FILE" ]; then
    if grep -q "__version__" "$INIT_FILE"; then
        sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$INIT_FILE"
        rm -f "$INIT_FILE.bak"
        echo "Updated version in $INIT_FILE"
    fi
fi

echo "Version bump complete!"
