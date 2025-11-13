# Copyright (c) Microsoft. All rights reserved.

import io
import logging
import multiprocessing as mp
from multiprocessing.queues import Queue
from typing import Any, Dict, List

import pytest

from agentlightning.logging import _to_level_value  # pyright: ignore[reportPrivateUsage]
from agentlightning.logging import (
    DATE_FORMAT,
    DEFAULT_FORMAT,
)


def _logging_worker(case: str, queue: Queue[Dict[str, Any]]):
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
            def __init__(self):
                super().__init__()
                self.records: List[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord):
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


def _run_case(case: str) -> Dict[str, Any]:
    """Helper to run a scenario in a spawn’ed process and fetch the result."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_logging_worker, args=(case, q))
    p.start()
    result = q.get(timeout=10)
    p.join(timeout=10)
    assert p.exitcode == 0
    return result


def test_to_level_value_int_and_str():
    # direct, no multiprocessing needed
    assert _to_level_value(logging.DEBUG) == logging.DEBUG
    assert _to_level_value("info") == logging.INFO
    assert _to_level_value("WARNING") == logging.WARNING

    with pytest.raises(ValueError):
        _to_level_value("not-a-level")


def test_setup_module_plain_console_spawn():
    result = _run_case("setup_module_plain_console")

    assert result["logger_name"] == "agentlightning.test"
    assert result["logger_level"] == logging.DEBUG

    # Console handler with plain formatter configured
    assert result["num_handlers"] == 1
    assert result["handler_class"].endswith("StreamHandler")
    assert result["handler_level"] == logging.DEBUG
    assert result["fmt"] == DEFAULT_FORMAT
    assert result["datefmt"] == DATE_FORMAT


def test_setup_module_color_rich_spawn():
    # Only run this test if rich is installed
    pytest.importorskip("rich")

    result = _run_case("setup_module_color_rich")

    assert result["logger_name"] == "agentlightning.rich"
    assert result["logger_level"] == logging.INFO
    assert result["num_handlers"] == 1
    # We can’t rely on full module path, just the class name
    assert result["handler_class"].endswith("RichHandler")


def test_setup_with_submodules_apply_to_and_capture_warnings_spawn():
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


def test_setup_with_console_and_extra_handler_spawn():
    result = _run_case("setup_with_console_and_extra_handler")

    # Level propagated to base logger
    assert result["base_level"] == logging.WARNING

    # Both console handler and extra handler should be attached
    assert result["num_handlers"] >= 2
    assert any(cls.endswith("StreamHandler") for cls in result["handler_classes"])
    assert result["has_extra"] is True
