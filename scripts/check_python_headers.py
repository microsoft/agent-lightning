# Copyright (c) Microsoft. All rights reserved.

"""Utility script to ensure Python files include the required copyright header."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_HEADER = "# Copyright (c) Microsoft. All rights reserved."
REPO_ROOT = Path(__file__).resolve().parent.parent


def iter_python_files() -> list[Path]:
    """Return a list of tracked Python files respecting .gitignore rules."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "*.py",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return [REPO_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    missing_header: list[str] = []
    missing_blank_line: list[str] = []

    for file_path in iter_python_files():
        try:
            with file_path.open("r", encoding="utf-8") as file:
                first_line = file.readline().rstrip("\r\n")
                second_line = file.readline()
        except OSError as exc:
            print(f"Failed to read {file_path}: {exc}", file=sys.stderr)
            return 1

        if first_line != REQUIRED_HEADER:
            missing_header.append(str(file_path.relative_to(REPO_ROOT)))
            continue

        # Second line should be either an EOF or a blank line
        if second_line and second_line.strip():
            missing_blank_line.append(str(file_path.relative_to(REPO_ROOT)))

    if missing_header:
        print("The following Python files are missing the required copyright header:")
        for path in missing_header:
            print(f" - {path}")
        print(
            "Run the appropriate script or add the header manually:",
            f"\n{REQUIRED_HEADER}",
        )

    if missing_blank_line:
        print("The following Python files are missing a blank line after the copyright header:")
        for path in missing_blank_line:
            print(f" - {path}")
        print("Ensure there is an empty line separating the header from the rest of the file.")

    if missing_header or missing_blank_line:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
