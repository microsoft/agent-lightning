# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import html
import os
from typing import Any, Dict


def define_env(env: Any):
    """Expose {{ src('path/to/file.py') }} to link to a file in the repo.

    Behavior:

        - Builds URL from repo_url + extra.source_commit in mkdocs.yml
        - Verifies that the file exists at build time
        - If file missing: logs a WARNING and returns a visible marker.
            With `mkdocs build --strict`, warnings become errors → build fails.

    Examples:
        ```
        [`apo_debug.py`]({{ src('examples/apo/apo_debug.py') }})
        ```

        or

        ```
        {{ src('examples/apo/apo_debug.py') }}
        ```
    """

    cfg: Dict[str, Any] = env.conf or {}
    repo_url = cfg.get("repo_url", "").rstrip("/")
    extra: Dict[str, Any] = cfg.get("extra", {}) or {}
    default_commit = extra.get("source_commit", "main")
    project_dir = env.project_dir

    if not repo_url:
        raise RuntimeError("repo_url must be set in mkdocs.yml for src() macro to work.")

    logger = getattr(env, "logger", None)

    def _warn(msg: str):
        if logger and hasattr(logger, "warning"):
            logger.warning(f"[macros:src] {msg}")
        else:
            print(f"[macros:src] WARNING: {msg}")

    def src(path: str, text: str | None = None, commit: str | None = None) -> str:
        commit = commit or default_commit
        abs_root = os.path.abspath(project_dir)
        abs_path = os.path.abspath(os.path.join(project_dir, path))

        # Prevent escaping project root
        if not abs_path.startswith(abs_root + os.sep) and abs_path != abs_root:
            raise ValueError(f"Invalid path outside project: {path}")

        # Build the GitHub tree URL (folder or file both work for our use case)
        url = f"{repo_url}/tree/{commit}/{path}"

        if not os.path.exists(abs_path):
            _warn(f"Source path not found: {path}. Rendering a visible broken-link marker.")
            label = html.escape(text or path)
            return (
                f'<span class="broken-source-link" title="Missing: {html.escape(url)}">' f"{label} (missing)" f"</span>"
            )

        if text:
            return f"[{text}]({url})"
        return url

    env.macro(src)
