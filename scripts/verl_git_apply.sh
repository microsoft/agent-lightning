#!/bin/bash

# --- Configuration ---
# The script will only run if the repo's HEAD is at this exact commit.
REQUIRED_COMMIT_ID="2dc3e0ebadb479bb3f2b48cfc7f28a3b70d5ce60"

# The directory containing the 'new' or 'patched' files to be applied.
# This is the source of truth for the final state.
PATCH_SOURCE_DIR="agentlightning/instrumentation/verl_patch"
# ---------------------

# --- Functions ---
usage() {
  echo "Usage: $0 <path_to_repo>"
  echo
  echo "Applies file additions and modifications from '$PATCH_SOURCE_DIR'"
  echo "to the specified repository path non-interactively."
  echo "The repository MUST be at commit $REQUIRED_COMMIT_ID."
  echo
  echo "Arguments:"
  echo "  <path_to_repo>   The path to the git repository to apply the patch to."
  exit 1
}

# --- Script Start ---

# The repository path is the first argument
VERL_REPO_PATH="$1"

# Check if repository path was provided
if [ -z "$VERL_REPO_PATH" ]; then
  echo "Error: Repository path is required."
  usage
fi

# --- Main Logic ---
echo "***************************************************"
echo "* Non-Interactive Repository Patch Apply Script  *"
echo "* (Add/Modify Operations Only)          *"
echo "***************************************************"
echo
echo "Target Repository: '$VERL_REPO_PATH'"
echo "Patch Source:      '$PATCH_SOURCE_DIR'"
echo "Required Commit:   '$REQUIRED_COMMIT_ID'"
echo

# --- Validations ---

# Expand the tilde (~) in the repo path to the user's home directory
VERL_REPO_PATH=$(eval echo "$VERL_REPO_PATH")

if [ ! -d "$VERL_REPO_PATH" ] || [ ! -d "$VERL_REPO_PATH/.git" ]; then
  echo "Error: Target repository path '$VERL_REPO_PATH' does not exist or is not a git repository."
  exit 1
fi

if [ ! -d "$PATCH_SOURCE_DIR" ]; then
  echo "Error: Patch source directory '$PATCH_SOURCE_DIR' does not exist."
  exit 1
fi

# --- PRE-FLIGHT CHECK: Verify Commit ID ---
echo "Verifying repository commit ID..."
current_commit=$(git -C "$VERL_REPO_PATH" rev-parse HEAD)

if [ "$current_commit" != "$REQUIRED_COMMIT_ID" ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! Critical Error: Incorrect repository state."
  echo "!! The script can only be applied when the repository's"
  echo "!! HEAD is at the required commit."
  echo "!! Expected: $REQUIRED_COMMIT_ID"
  echo "!! Found:    $current_commit"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
else
  echo "Success. Repository is at the correct commit."
fi

# --- Apply Additions and Modifications ---
echo
echo "Applying file additions and modifications..."
find "$PATCH_SOURCE_DIR" -type f -print0 | while IFS= read -r -d $'\0' source_file; do
    relative_path="${source_file#$PATCH_SOURCE_DIR/}"
    destination_file="$VERL_REPO_PATH/$relative_path"

    # Ensure the destination directory exists before copying
    mkdir -p "$(dirname "$destination_file")"
    
    # Copy the file, overwriting the destination
    cp -v "$source_file" "$destination_file"
done

echo
echo "Patch application complete."
echo "Review the changes in '$VERL_REPO_PATH' with 'git status' or 'git diff'."
