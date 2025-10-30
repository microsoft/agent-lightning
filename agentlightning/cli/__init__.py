# Copyright (c) 2024, Microsoft Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The command line interface for Agent Lightning."""

import argparse
import sys

from agentlightning.logging import setup_logger
from agentlightning.settings import get_settings

# Static imports of allowed command modules
try:
    from agentlightning.cli.commands import chat
except ImportError:
    chat = None

try:
    from agentlightning.cli.commands import config
except ImportError:
    config = None

# Command registry with explicit mapping
COMMAND_REGISTRY = {
    "chat": chat,
    "config": config,
}


def main():
    """The main entry point for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Agent Lightning CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register commands from the static registry
    for command_name, module in COMMAND_REGISTRY.items():
        if module is not None and hasattr(module, "register"):
            module.register(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        # setup logger
        settings = get_settings()
        setup_logger(settings.log_level, settings.log_file)
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main())
