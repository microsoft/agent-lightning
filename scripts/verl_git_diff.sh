#!/bin/bash

# --- Configuration ---
# The commit hash to compare against.
COMMIT_HASH="2dc3e0ebadb479bb3f2b48cfc7f28a3b70d5ce60"
# The absolute or relative path to the git repository.
VERL_REPO_PATH=${VERL_REPO_PATH:-"~/verl"}
# The subfolder within the repository to limit the diff to.
SUBFOLDER="verl"
# The output directory for files from the old commit.
OLD_VERSION_DIR="agentlightning/instrumentation/verl_old"
# The output directory for files from the current version (HEAD).
CURRENT_VERSION_DIR="agentlightning/instrumentation/verl_patch"
# ---------------------

# --- Script Start ---

# Store the original directory where the script was run
ORIGINAL_DIR=$(pwd)

# Expand the tilde (~) in the repo path to the user's home directory
VERL_REPO_PATH=$(eval echo "$VERL_REPO_PATH")

# --- Validations ---
if [ ! -d "$VERL_REPO_PATH" ]; then
  echo "Error: Repository path '$VERL_REPO_PATH' does not exist."
  exit 1
fi

if [ ! -d "$VERL_REPO_PATH/.git" ]; then
  echo "Error: Directory '$VERL_REPO_PATH' is not a git repository."
  exit 1
fi

# Change to the repository directory
cd "$VERL_REPO_PATH" || exit

# --- Main Logic ---
echo "Creating output directories in '$ORIGINAL_DIR'..."
# Use absolute paths for output directories to ensure they are created relative to the original location
mkdir -p "$ORIGINAL_DIR/$OLD_VERSION_DIR"
mkdir -p "$ORIGINAL_DIR/$CURRENT_VERSION_DIR"

echo "Finding differences between $COMMIT_HASH and HEAD in '$SUBFOLDER'..."

# Get the list of changed files relative to the repo root
changed_files=$(git diff --name-only "$COMMIT_HASH" HEAD -- "$SUBFOLDER")

if [ -z "$changed_files" ]; then
  echo "No differences found in the specified subfolder."
  cd "$ORIGINAL_DIR" # Go back to the original directory before exiting
  exit 0
fi

echo "Exporting files..."

while read -r file; do
  echo "  -> $file"

  # Define absolute paths for destination files
  dest_old="$ORIGINAL_DIR/$OLD_VERSION_DIR/$file"
  dest_current="$ORIGINAL_DIR/$CURRENT_VERSION_DIR/$file"

  # Create the directory structure within the output directories
  mkdir -p "$(dirname "$dest_old")"
  mkdir -p "$(dirname "$dest_current")"

  # Get the file from the specific commit and redirect output to the absolute path
  git show "$COMMIT_HASH:$file" > "$dest_old" 2>/dev/null || echo "    - Note: File did not exist in the old commit."

  # Copy the file from the current working directory (which is VERL_REPO_PATH) to the absolute path
  if [ -f "$file" ]; then
    cp "$file" "$dest_current"
  else
    echo "    - Note: File is deleted in the current version."
  fi

done <<< "$changed_files"

# Change back to the original directory
cd "$ORIGINAL_DIR"

echo
echo "Done. You can now compare the contents of '$OLD_VERSION_DIR' and '$CURRENT_VERSION_DIR'."
