"""Logging configuration for agl-lite processes.

Configures structlog with dual output:
- stdout: ConsoleRenderer (human-friendly, colored)
- file:   JSONRenderer (JSON Lines) — only when log_dir is set

Uses ProcessorFormatter as a bridge so stdlib logging records
(uvicorn, third-party libraries) also route through structlog processors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(log_dir: str | None, log_level: str, component: str) -> None:
    """Configure structlog for a named component process.

    Args:
        log_dir:   Directory for log files. If None, file output is disabled.
        log_level: stdlib log level name (DEBUG / INFO / WARNING / ERROR).
        component: Process name — log file is written to log_dir/<component>.log.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Processors shared by both handlers and applied to all records
    # (both structlog-native and stdlib foreign records via foreign_pre_chain).
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # remove any default handlers before adding ours

    # ── stdout: human-friendly ───────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    ))
    root.addHandler(console_handler)

    # ── file: JSON Lines ─────────────────────────────────────────────
    if log_dir:
        log_path = Path(log_dir) / f"{component}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        ))
        root.addHandler(file_handler)
