# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
import platform
from logging.config import dictConfig
from typing import Any, Dict, Optional

__all__ = ["configure_logger"]


def configure_logger(level: int = logging.INFO, name: str = "agentlightning") -> logging.Logger:
    """Create or reset a namespaced logger with a consistent console format.

    This helper clears any previously attached handlers before binding a single
    `StreamHandler` that writes to standard output. The resulting logger does
    not propagate to the root logger, preventing duplicate log emission when
    applications compose multiple logging configurations.

    Args:
        level: Logging level applied both to the logger and the installed
            handler. Defaults to `logging.INFO`.
        name: Dotted path for the logger instance. Defaults to
            `"agentlightning"`.

    Returns:
        Configured logger instance ready for immediate use.

    Examples:
        ```python
        from agentlightning import configure_logger

        logger = configure_logger(level=logging.INFO)
        logger.info("agent-lightning is ready!")
        ```
    """

    return setup_module(level=level, name=name, console=True, color=True, propagate=False)


DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s"
DATE_FORMAT = "%H:%M:%S"


def _to_level_value(lvl: int | str) -> int:
    if isinstance(lvl, int):
        return lvl
    val = getattr(logging, str(lvl).upper(), None)
    if val is None:
        raise ValueError(f"Invalid log level: {lvl}")
    return val


def setup(
    level: int | str = "INFO",
    *,
    console: bool = True,
    color: bool | Dict[str, Any] = True,
    propagate: bool = False,
    disable_existing_loggers: bool = False,
    capture_warnings: bool = False,
    submodule_levels: Optional[dict[str, int | str]] = None,
    extra_handlers: Optional[list[logging.Handler]] = None,
    formatter: Optional[logging.Formatter] = None,
    apply_to: Optional[list[str]] = None,
) -> None:
    """Configures logging for the `agentlightning` logger hierarchy.

    This function provides a one-stop setup utility for configuring the
    `agentlightning` root logger and optionally its submodules or external
    loggers. It supports console logging, colored rich output, per-submodule
    log levels, and optional handler/formatter injection.

    The setup is intentionally isolated: it does not modify the global root
    logger or loggers belonging to other libraries unless explicitly directed
    via `apply_to`.

    Args:
        level:
            Logging level for the base `agentlightning` logger. Accepts either
            an integer (e.g., `logging.DEBUG`) or a string level name
            (e.g., `"INFO"`). Defaults to `"INFO"`.
        console:
            Whether to attach a console handler to the logger. Defaults to
            `True`.
        color:
            Enables rich-formatted output using `RichHandler` when `True`
            or a configuration dict. If `False`, a plain text formatter is
            used instead. Defaults to `True`.
        propagate:
            Whether `agentlightning` logs should propagate to ancestor
            loggers. Defaults to `False`.
        disable_existing_loggers:
            Passed to `logging.config.dictConfig`. If `True`, disables all
            existing configured loggers before applying this configuration.
            Defaults to `False`.
        capture_warnings:
            If `True`, redirects Python `warnings` emitted via the `warnings`
            module into the logging system. Defaults to `False`.
        submodule_levels:
            Mapping of submodule logger names to logging levels. If a specified
            submodule level is more verbose than the base level, a warning is emitted.
        extra_handlers:
            A list of user-provided handlers to attach to the `agentlightning` logger.
            Handlers are added idempotently; duplicates are not reattached.
        formatter:
            A formatter to apply to any handler under `agentlightning` that does not
            already have one assigned. Useful for customizing output without overwriting
            formatters on custom handlers.
        apply_to:
            A list of additional logger names to configure identically to
            `agentlightning`. Their handlers are replaced with copies of the base
            handlers, and propagation is disabled to avoid duplicate log emission.

    Notes:
        * On Windows, this function forces UTF-8 mode in the console to prevent
          issues with rich output or special characters.
        * Submodule loggers can generate records below the handler's emission
          threshold. Whether such records appear depends on both the logger's
          level and the handler's level.
        * `apply_to` loggers inherit the same handlers but do not propagate
          upward, yielding isolated, consistent behavior.

    Examples:
        Basic setup:

        >>> setup()

        Enabling debug mode with no color:

        >>> setup(level="DEBUG", color=False)

        Overriding specific submodule levels:

        >>> setup(submodule_levels={"agentlightning.io": "DEBUG"})

        Attaching an additional file handler:

        >>> fh = logging.FileHandler("app.log")
        >>> setup(extra_handlers=[fh])
    """
    # Ensure UTF-8 encoding on Windows consoles
    # Note: This change does not fully represent support for execution under the windown system.
    # It only fixes console printing issues caused by special characters.
    # TODO: More comprehensive Windows support may be needed in the future.
    if platform.system() == "Windows":
        os.environ["PYTHONUTF8"] = "1"

    base_logger = setup_module(
        level,
        name="agentlightning",
        console=console,
        color=color,
        propagate=propagate,
        disable_existing_loggers=disable_existing_loggers,
    )

    # Apply user-provided formatter (only to handlers without one,
    # so we don't clobber custom extra_handlers)
    if formatter is not None:
        for h in base_logger.handlers:
            if h.formatter is None:
                h.setFormatter(formatter)

    # Attach user-provided handler(s) if any, idempotently
    if extra_handlers:
        for h in extra_handlers:
            if h not in base_logger.handlers:
                base_logger.addHandler(h)

    # Per-submodule levels
    if submodule_levels:
        base_level = base_logger.level
        for name, lvl in submodule_levels.items():
            sub_level = _to_level_value(lvl)

            # Emit a warning if submodule level is lower (more verbose) than the global/base level
            if sub_level < base_level:
                base_logger.warning(
                    "Submodule logger '%s' level %s (%s) is more verbose than base "
                    "logger level %s (%s). Records below the base level may still be "
                    "filtered out by handlers depending on their own levels.",
                    name,
                    lvl,
                    sub_level,
                    logging.getLevelName(base_level),
                    base_level,
                )

            # The logger will *create* records down to the logger's level, but a handler
            # with a higher level will still drop anything below its own threshold.
            # Effective emission is gated by both: record.level >= logger.level AND handler.level.
            logging.getLogger(name).setLevel(lvl)

    # Optionally apply the same handler setup to other loggers outside this module
    if apply_to:
        for name in apply_to:
            lg = logging.getLogger(name)
            # This removes any existing handlers so we don't duplicate output
            # and ensures these loggers share exactly the same handlers as base_logger.
            lg.handlers.clear()
            for h in base_logger.handlers:
                lg.addHandler(h)
            lg.setLevel(base_logger.level)
            # We've attached handlers directly to these loggers; if propagate
            # stayed True, records would bubble up to ancestor loggers and could be
            # emitted twice (here and on the parent/root). Setting False isolates them.
            lg.propagate = False

    # Optionally capture warnings
    if capture_warnings:
        logging.captureWarnings(True)


def setup_module(
    level: int | str = "INFO",
    *,
    name: str = "agentlightning",
    console: bool = True,
    color: bool | Dict[str, Any] = True,
    propagate: bool = False,
    disable_existing_loggers: bool = False,
) -> logging.Logger:
    """Initializes and returns the base logger for `agentlightning`.

    This function constructs and applies a `dictConfig` configuration for the
    logger hierarchy rooted at `name`. It supports either rich console
    formatting (via `RichHandler`) or plain text formatting, based on the
    `color` argument.

    Unlike `setup()`, this function configures only a single logger namespace
    and does not attach extra handlers or submodule levels. It is primarily used
    internally by `setup()` but is also suitable for direct integration in
    custom logging workflows.
    """
    root_cfg: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": disable_existing_loggers,
        "loggers": {
            name: {
                "handlers": [],
                "level": level,
                "propagate": propagate,
            }
        },
        "handlers": {},
        "formatters": {},
    }

    # Choose formatter / handler definition
    if color is not False:
        fmt_name = "rich"
        if isinstance(color, dict):
            rich_handler_config = color
        else:
            rich_handler_config: Dict[str, Any] = {
                "rich_tracebacks": True,
                "markup": False,
                "show_time": True,
                "show_path": True,
            }
        root_cfg["handlers"]["console"] = {
            "class": "rich.logging.RichHandler",
            "level": level,
            **rich_handler_config,
        }
        # RichHandler manages its own style; keep formatter None
    else:
        fmt_name = "plain"
        root_cfg["formatters"][fmt_name] = {
            "format": DEFAULT_FORMAT,
            "datefmt": DATE_FORMAT,
        }

        if console:
            root_cfg["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": fmt_name,
            }

    # Attach selected handlers to agentlightning
    handler_names = list(root_cfg["handlers"].keys())
    root_cfg["loggers"][name]["handlers"] = handler_names

    # Apply dictConfig (this resets the logger handlers)
    dictConfig(root_cfg)

    return logging.getLogger(name)


# I'm not sure about the safety and correctness of this method. Please write some type annotated functional-style pytests to test the logger.
