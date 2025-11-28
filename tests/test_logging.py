# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import io
import logging
import multiprocessing as mp
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentlightning.logging import _to_level_value  # pyright: ignore[reportPrivateUsage]
from agentlightning.logging import (
    DATE_FORMAT,
    DEFAULT_FORMAT,
)


def _logging_worker(case: str, queue: Queue[Dict[str, Any]]) -> None:
    """
    Runs in a separate process using spawn. It performs a specific logging
    configuration scenario and returns a summary dict via the queue.
    """
    import logging
    import warnings

    # Re-import inside the subprocess so everything is picklable & isolated
    from agentlightning.logging import (
        setup,
        setup_module,
    )

    if case == "setup_module_plain_console":
        logger = setup_module(
            level="DEBUG",
            name="agentlightning.test",
            console=True,
            color=False,
            propagate=False,
        )

        handlers = logger.handlers
        handler = handlers[0] if handlers else None
        fmt = handler.formatter._fmt if handler and handler.formatter else None
        datefmt = handler.formatter.datefmt if handler and handler.formatter else None

        queue.put(
            {
                "logger_name": logger.name,
                "logger_level": logger.level,
                "num_handlers": len(handlers),
                "handler_class": handler.__class__.__name__ if handler else None,
                "handler_level": handler.level if handler else None,
                "fmt": fmt,
                "datefmt": datefmt,
            }
        )

    elif case == "setup_module_color_rich":
        # Rich variant: color=True uses RichHandler
        logger = setup_module(
            level="INFO",
            name="agentlightning.rich",
            console=True,
            color=True,
            propagate=False,
        )
        handlers = logger.handlers
        handler = handlers[0] if handlers else None

        queue.put(
            {
                "logger_name": logger.name,
                "logger_level": logger.level,
                "num_handlers": len(handlers),
                "handler_class": handler.__class__.__name__ if handler else None,
                "handler_has_formatter": handler.formatter is not None if handler else None,
            }
        )

    elif case == "setup_with_submodules_apply_to_capture_warnings":
        # Extra handler to attach via extra_handlers
        stream = io.StringIO()
        stream_handler = logging.StreamHandler(stream)

        setup(
            level="INFO",
            console=False,
            color=False,
            propagate=False,
            disable_existing_loggers=False,
            capture_warnings=True,
            submodule_levels={"agentlightning.io": "DEBUG"},
            extra_handlers=[stream_handler],
            apply_to=["external"],
        )

        base = logging.getLogger("agentlightning")
        sub = logging.getLogger("agentlightning.io")
        ext = logging.getLogger("external")

        # Capture warnings via logging after capture_warnings=True
        class ListHandler(logging.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.records: List[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord) -> None:
                self.records.append(record)

        lh = ListHandler()
        wlog = logging.getLogger("py.warnings")
        wlog.handlers.clear()
        wlog.addHandler(lh)
        wlog.setLevel(logging.WARNING)
        wlog.propagate = False

        warnings.warn("from warnings", UserWarning)

        queue.put(
            {
                "base_level": base.level,
                "base_num_handlers": len(base.handlers),
                "extra_in_base": stream_handler in base.handlers,
                "sub_level": sub.level,
                "ext_level": ext.level,
                "ext_handlers_same": base.handlers == ext.handlers,
                "ext_propagate": ext.propagate,
                "warnings_logged": len(lh.records),
            }
        )

    elif case == "setup_with_console_and_extra_handler":
        # Console + extra handler combination to test handler attachment
        stream = io.StringIO()
        extra_handler = logging.StreamHandler(stream)

        setup(
            level="WARNING",
            console=True,
            color=False,
            propagate=False,
            extra_handlers=[extra_handler],
        )

        base = logging.getLogger("agentlightning")
        handler_classes = [h.__class__.__name__ for h in base.handlers]
        has_extra = extra_handler in base.handlers

        queue.put(
            {
                "base_level": base.level,
                "num_handlers": len(base.handlers),
                "handler_classes": handler_classes,
                "has_extra": has_extra,
            }
        )

    else:
        queue.put({})


def _logging_worker_files_string(queue: Queue[Dict[str, Any]], base_dir: str) -> None:
    """
    Runs in a separate spawned process and configures logging with a single
    files=str path. Returns information about the attached FileHandler.
    """
    import logging
    import os

    from agentlightning.logging import setup

    log_path = os.path.join(base_dir, "logs", "agent.log")

    setup(
        level="INFO",
        console=False,
        color=False,
        propagate=False,
        files=log_path,
    )

    base = logging.getLogger("agentlightning")
    file_handlers = [h for h in base.handlers if isinstance(h, logging.FileHandler)]
    fh = file_handlers[0] if file_handlers else None

    fmt = fh.formatter._fmt if fh and fh.formatter else None
    datefmt = fh.formatter.datefmt if fh and fh.formatter else None

    queue.put(
        {
            "logger_level": base.level,
            "num_handlers": len(base.handlers),
            "num_file_handlers": len(file_handlers),
            "file_base": fh.baseFilename if fh else None,
            "file_level": fh.level if fh else None,
            "fmt": fmt,
            "datefmt": datefmt,
        }
    )


def _logging_worker_files_mapping(queue: Queue[Dict[str, Any]], base_dir: str) -> None:
    """
    Runs in a separate spawned process and configures logging with a files=dict
    mapping, then calls setup twice to verify idempotent FileHandler attachment.
    """
    import logging
    import os

    from agentlightning.logging import setup

    base_log = os.path.join(base_dir, "agent.log")
    external_log = os.path.join(base_dir, "external.log")

    files_mapping: Dict[str, str] = {
        "agentlightning": base_log,
        "external": external_log,
    }

    def file_handlers(logger: logging.Logger) -> list[logging.FileHandler]:
        return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]

    # First setup call
    setup(
        level="DEBUG",
        console=False,
        color=False,
        propagate=False,
        files=files_mapping,
    )

    base_logger = logging.getLogger("agentlightning")
    ext_logger = logging.getLogger("external")

    base_fh_first = file_handlers(base_logger)
    ext_fh_first = file_handlers(ext_logger)

    # Second setup call with the same mapping should not add duplicate FileHandlers
    setup(
        level="DEBUG",
        console=False,
        color=False,
        propagate=False,
        files=files_mapping,
    )

    base_fh_second = file_handlers(base_logger)
    ext_fh_second = file_handlers(ext_logger)

    queue.put(
        {
            "base_level": base_logger.level,
            "ext_level": ext_logger.getEffectiveLevel(),
            "base_first_count": len(base_fh_first),
            "ext_first_count": len(ext_fh_first),
            "base_second_count": len(base_fh_second),
            "ext_second_count": len(ext_fh_second),
            "base_file_first": base_fh_first[0].baseFilename if base_fh_first else None,
            "ext_file_first": ext_fh_first[0].baseFilename if ext_fh_first else None,
            "base_file_second": base_fh_second[0].baseFilename if base_fh_second else None,
            "ext_file_second": ext_fh_second[0].baseFilename if ext_fh_second else None,
            # For sanity: capture handler levels as well
            "base_handler_level": base_fh_first[0].level if base_fh_first else None,
            "ext_handler_level": ext_fh_first[0].level if ext_fh_first else None,
        }
    )


def _run_case(case: str) -> Dict[str, Any]:
    """Helper to run a scenario in a spawn’ed process and fetch the result."""
    ctx = mp.get_context("spawn")
    q: Queue[Dict[str, Any]] = ctx.Queue()
    p = ctx.Process(target=_logging_worker, args=(case, q))
    p.start()
    result = q.get(timeout=10)
    p.join(timeout=10)
    assert p.exitcode == 0
    return result


def test_to_level_value_int_and_str() -> None:
    # direct, no multiprocessing needed
    assert _to_level_value(logging.DEBUG) == logging.DEBUG
    assert _to_level_value("info") == logging.INFO
    assert _to_level_value("WARNING") == logging.WARNING

    with pytest.raises(ValueError):
        _to_level_value("not-a-level")


def test_setup_module_plain_console_spawn() -> None:
    result = _run_case("setup_module_plain_console")

    assert result["logger_name"] == "agentlightning.test"
    assert result["logger_level"] == logging.DEBUG

    # Console handler with plain formatter configured
    assert result["num_handlers"] == 1
    assert result["handler_class"].endswith("StreamHandler")
    assert result["handler_level"] == logging.DEBUG
    assert result["fmt"] == DEFAULT_FORMAT
    assert result["datefmt"] == DATE_FORMAT


def test_setup_module_color_rich_spawn() -> None:
    # Only run this test if rich is installed
    pytest.importorskip("rich")

    result = _run_case("setup_module_color_rich")

    assert result["logger_name"] == "agentlightning.rich"
    assert result["logger_level"] == logging.INFO
    assert result["num_handlers"] == 1
    # We can’t rely on full module path, just the class name
    assert result["handler_class"].endswith("RichHandler")


def test_setup_with_submodules_apply_to_and_capture_warnings_spawn() -> None:
    result = _run_case("setup_with_submodules_apply_to_capture_warnings")

    # Base logger level and handler attachment
    assert result["base_level"] == logging.INFO
    assert result["base_num_handlers"] >= 1
    assert result["extra_in_base"] is True

    # Submodule level overridden
    assert result["sub_level"] == logging.DEBUG

    # apply_to logger mirrors base handlers & level, propagation disabled
    assert result["ext_level"] == logging.INFO
    assert result["ext_handlers_same"] is True
    assert result["ext_propagate"] is False

    # capture_warnings=True causes warnings.warn to go through logging
    assert result["warnings_logged"] >= 1


def test_setup_with_console_and_extra_handler_spawn() -> None:
    result = _run_case("setup_with_console_and_extra_handler")

    # Level propagated to base logger
    assert result["base_level"] == logging.WARNING

    # Both console handler and extra handler should be attached
    assert result["num_handlers"] >= 2
    assert any(cls.endswith("StreamHandler") for cls in result["handler_classes"])
    assert result["has_extra"] is True


def test_setup_files_string_spawn(tmp_path: Path) -> None:
    """
    Verifies that passing files as a string attaches a single FileHandler with
    the expected level and default formatter in a spawned process.
    """
    ctx = mp.get_context("spawn")
    q: Queue[Dict[str, Any]] = ctx.Queue()
    p = ctx.Process(target=_logging_worker_files_string, args=(q, str(tmp_path)))
    p.start()
    result = q.get(timeout=10)
    p.join(timeout=10)
    assert p.exitcode == 0

    assert result["logger_level"] == logging.INFO
    # We expect at least one handler and exactly one FileHandler
    assert result["num_handlers"] >= 1
    assert result["num_file_handlers"] == 1

    # Filename should be inside the tmp_path tree
    assert str(tmp_path) in result["file_base"]
    # FileHandler uses the base logger level
    assert result["file_level"] == logging.INFO

    # Default formatter applied by _ensure_file_handler
    assert result["fmt"] == DEFAULT_FORMAT
    assert result["datefmt"] == DATE_FORMAT


def test_setup_files_mapping_spawn(tmp_path: Path) -> None:
    """
    Verifies that passing files as a mapping attaches FileHandlers to each
    logger and that calling setup twice does not create duplicate handlers.
    """
    ctx = mp.get_context("spawn")
    q: Queue[Dict[str, Any]] = ctx.Queue()
    p = ctx.Process(target=_logging_worker_files_mapping, args=(q, str(tmp_path)))
    p.start()
    result = q.get(timeout=10)
    p.join(timeout=10)
    assert p.exitcode == 0

    # Base logger level is DEBUG
    assert result["base_level"] == logging.DEBUG

    # External's effective level is WARNING (inherited from root)
    assert result["ext_level"] == logging.WARNING

    # First setup: one FileHandler per logger
    assert result["base_first_count"] == 1
    assert result["ext_first_count"] == 1

    # Second setup: still one FileHandler per logger (idempotence)
    assert result["base_second_count"] == 1
    assert result["ext_second_count"] == 1

    # File paths are stable across calls
    assert result["base_file_first"] == result["base_file_second"]
    assert result["ext_file_first"] == result["ext_file_second"]

    # Paths should live under tmp_path
    assert str(tmp_path) in result["base_file_first"]
    assert str(tmp_path) in result["ext_file_first"]

    # Handler levels:
    # - base handler uses the base logger level (DEBUG)
    # - external handler uses external's effective level at creation (WARNING)
    assert result["base_handler_level"] == logging.DEBUG
    assert result["ext_handler_level"] == logging.WARNING
