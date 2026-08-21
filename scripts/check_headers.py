# Copyright (c) Microsoft. All rights reserved.

"""Ensure tracked Python files include the required copyright header."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HEADER = "# Copyright (c) Microsoft. All rights reserved."
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = (".py", ".pyi", ".pyw")
EXCLUDED_PREFIXES = ("examples/llm-in-sandbox/vendor/",)


def iter_source_files() -> list[Path]:
    """Return tracked and untracked Python source files."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *(f"*{suffix}" for suffix in SOURCE_SUFFIXES),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    paths = (line.strip() for line in result.stdout.splitlines())
    return [
        REPO_ROOT / path for path in paths if path and not any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    ]


def main() -> int:
    missing_header: list[str] = []
    missing_blank_line: list[str] = []

    for file_path in iter_source_files():
        if not file_path.exists():
            continue

        try:
            with file_path.open("r", encoding="utf-8") as file:
                first_line = file.readline().rstrip("\r\n")
                header_line = file.readline().rstrip("\r\n") if first_line.startswith("#!") else first_line
                following_line = file.readline()
        except OSError as exc:
            print(f"Failed to read {file_path}: {exc}", file=sys.stderr)
            return 1

        relative_path = str(file_path.relative_to(REPO_ROOT))
        if header_line != HEADER:
            missing_header.append(relative_path)
            continue
        if following_line and following_line.strip():
            missing_blank_line.append(relative_path)

    if missing_header:
        print("The following files are missing the required copyright header:")
        for path in missing_header:
            print(f" - {path}")
        print(f"Add this header after any shebang:\n{HEADER}")

    if missing_blank_line:
        print("The following files are missing a blank line after the copyright header:")
        for path in missing_blank_line:
            print(f" - {path}")

    return 1 if missing_header or missing_blank_line else 0


if __name__ == "__main__":
    sys.exit(main())
