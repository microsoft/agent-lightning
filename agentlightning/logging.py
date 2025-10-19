# Copyright (c) Microsoft. All rights reserved.

import logging

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

    logger = logging.getLogger(name)
    logger.handlers.clear()  # clear existing handlers

    # log to stdout
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging
    return logger
