# Copyright (c) Microsoft. All rights reserved.

import logging

from agentlightning.logging import setup


def another_program():
    logger = logging.getLogger("agentlightning.test")
    logger.debug("This is a debug log message.")
    logger.info("This is an info log message.")
    logger.warning("This is a warning log message.")
    logger.error("This is an error log message.")
    logger.critical("This is a critical log message.")


def basic_program():
    setup()
    another_program()


if __name__ == "__main__":
    basic_program()
