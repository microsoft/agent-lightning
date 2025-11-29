# Copyright (c) Microsoft. All rights reserved.

"""Ensure tracked source files include the required copyright header."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HEADER_TEXT = "Copyright (c) Microsoft. All rights reserved."
REPO_ROOT = Path(__file__).resolve().parent.parent

COMMENT_PREFIX_BY_SUFFIX: dict[str, str] = {
    ".py": "#",
    ".pyi": "#",
    ".pyw": "#",
    ".js": "//",
    ".jsx": "//",
    ".ts": "//",
    ".tsx": "//",
    ".mjs": "//",
    ".mts": "//",
    ".cjs": "//",
    ".cts": "//",
}
REQUIRED_HEADER_BY_SUFFIX = {
    suffix: f"{prefix} {HEADER_TEXT}" if not prefix.endswith(" ") else f"{prefix}{HEADER_TEXT}"
    for suffix, prefix in COMMENT_PREFIX_BY_SUFFIX.items()
}


def iter_source_files() -> list[Path]:
    """Return tracked source files matching supported extensions."""
    if not REQUIRED_HEADER_BY_SUFFIX:
        return []

    pathspecs = [f"*{suffix}" for suffix in sorted(REQUIRED_HEADER_BY_SUFFIX)]
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *pathspecs,
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

    for file_path in iter_source_files():
        expected_header = REQUIRED_HEADER_BY_SUFFIX.get(file_path.suffix.lower())
        if expected_header is None:
            continue

        if not file_path.exists():
            continue

        try:
            with file_path.open("r", encoding="utf-8") as file:
                first_line = file.readline().rstrip("\r\n")
                second_line = file.readline()
        except OSError as exc:
            print(f"Failed to read {file_path}: {exc}", file=sys.stderr)
            return 1

        if first_line != expected_header:
            missing_header.append(str(file_path.relative_to(REPO_ROOT)))
            continue

        # Second line should be either an EOF or a blank line
        if second_line and second_line.strip():
            missing_blank_line.append(str(file_path.relative_to(REPO_ROOT)))

    if missing_header:
        print("The following files are missing the required copyright header:")
        for path in missing_header:
            print(f" - {path}")
        header_examples = "\n".join(sorted(set(REQUIRED_HEADER_BY_SUFFIX.values())))
        print(f"Run the appropriate script or add the header manually:\n{header_examples}")

    if missing_blank_line:
        print("The following files are missing a blank line after the copyright header:")
        for path in missing_blank_line:
            print(f" - {path}")
        print("Ensure there is an empty line separating the header from the rest of the file.")

    if missing_header or missing_blank_line:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
