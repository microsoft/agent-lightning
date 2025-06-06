#!/bin/bash

# --- Configuration ---
COMMIT_HASH="2dc3e0ebadb479bb3f2b48cfc7f28a3b70d5ce60"
VERL_REPO_PATH="~/verl"
SUBFOLDER="verl"
OLD_VERSION_DIR="agentlightning/instrumentation/verl_old"
CURRENT_VERSION_DIR="agentlightning/instrumentation/verl_patch"
# ---------------------

# --- Script Start ---
if [ "$COMMIT_HASH" == "<your-commit-hash>" ] || [ "$SUBFOLDER" == "<path/to/your/subfolder>" ]; then
  echo "Please edit the script and set the COMMIT_HASH and SUBFOLDER variables."
  exit 1
fi

echo "Creating output directories..."
mkdir -p "$OLD_VERSION_DIR"
mkdir -p "$CURRENT_VERSION_DIR"

echo "Finding differences between $COMMIT_HASH and HEAD in '$SUBFOLDER'..."

# Get the list of changed files
changed_files=$(git diff --name-only "$COMMIT_HASH" HEAD -- "$SUBFOLDER")

if [ -z "$changed_files" ]; then
  echo "No differences found in the specified subfolder."
  exit 0
fi

echo "Exporting files..."

while read -r file; do
  echo "  -> $file"

  # Create the directory structure in the output directories
  mkdir -p "$(dirname "$OLD_VERSION_DIR/$file")"
  mkdir -p "$(dirname "$CURRENT_VERSION_DIR/$file")"

  # Get the file from the specific commit
  git show "$COMMIT_HASH:$file" > "$OLD_VERSION_DIR/$file" 2>/dev/null || echo "    - Note: File did not exist in the old commit."

  # Copy the file from the current working directory
  if [ -f "$file" ]; then
    cp "$file" "$CURRENT_VERSION_DIR/$file"
  else
    echo "    - Note: File is deleted in the current version."
  fi

done <<< "$changed_files"

# Restore the state of any files that were checked out
git checkout HEAD -- "$SUBFOLDER" >/dev/null 2>&1

echo "Done. You can now compare the contents of '$OLD_VERSION_DIR' and '$CURRENT_VERSION_DIR'."