# mypy: allow-untyped-defs, allow-incomplete-defs, allow-untyped-calls
# mypy: no-warn-return-any, allow-any-generics

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from .. import util
from ..util import compat
from ..util.pyfiles import _preserving_path_as_str

if TYPE_CHECKING:
    from ..config import PostWriteHookConfig

REVISION_SCRIPT_TOKEN = "REVISION_SCRIPT_FILENAME"

_registry: dict = {}


def register(name: str) -> Callable:
    """A function decorator that will register that function as a write hook.

    See the documentation linked below for an example.

    .. seealso::

        :ref:`post_write_hooks_custom`


    """

    def decorate(fn):
        _registry[name] = fn
        return fn

    return decorate


def _invoke(
    name: str,
    revision_path: Union[str, os.PathLike[str]],
    options: PostWriteHookConfig,
) -> Any:
    """Invokes the formatter registered for the given name.

    :param name: The name of a formatter in the registry
    :param revision: string path to the revision file
    :param options: A dict containing kwargs passed to the
        specified formatter.
    :raises: :class:`alembic.util.CommandError`
    """
    revision_path = _preserving_path_as_str(revision_path)
    try:
        hook = _registry[name]
    except KeyError as ke:
        raise util.CommandError(
            f"No formatter with name '{name}' registered"
        ) from ke
    else:
        return hook(revision_path, options)


def _run_hooks(
    path: Union[str, os.PathLike[str]], hooks: list[PostWriteHookConfig]
) -> None:
    """Invoke hooks for a generated revision."""

    for hook in hooks:
        name = hook["_hook_name"]
        try:
            type_ = hook["type"]
        except KeyError as ke:
            raise util.CommandError(
                f"Key '{name}.type' (or 'type' in toml) is required "
                f"for post write hook {name!r}"
            ) from ke
        else:
            with util.status(
                f"Running post write hook {name!r}", newline=True
            ):
                _invoke(type_, path, hook)


def _parse_cmdline_options(cmdline_options_str: str, path: str) -> List[str]:
    """Parse options from a string into a list.

    Also substitutes the revision script token with the actual filename of
    the revision script.

    If the revision script token doesn't occur in the options string, it is
    automatically prepended.
    """
    if REVISION_SCRIPT_TOKEN not in cmdline_options_str:
        cmdline_options_str = REVISION_SCRIPT_TOKEN + " " + cmdline_options_str
    cmdline_options_list = shlex.split(
        cmdline_options_str, posix=compat.is_posix
    )
    cmdline_options_list = [
        option.replace(REVISION_SCRIPT_TOKEN, path)
        for option in cmdline_options_list
    ]
    return cmdline_options_list


def _get_required_option(options: dict, name: str) -> str:
    try:
        return options[name]
    except KeyError as ke:
        raise util.CommandError(
            f"Key {options['_hook_name']}.{name} is required for post "
            f"write hook {options['_hook_name']!r}"
        ) from ke


def _run_hook(
    path: str, options: dict, ignore_output: bool, command: List[str]
) -> None:
    cwd: Optional[str] = options.get("cwd", None)
    cmdline_options_str = options.get("options", "")
    cmdline_options_list = _parse_cmdline_options(cmdline_options_str, path)

    kw: Dict[str, Any] = {}
    if ignore_output:
        kw["stdout"] = kw["stderr"] = subprocess.DEVNULL

    subprocess.run([*command, *cmdline_options_list], cwd=cwd, **kw)


@register("console_scripts")
def console_scripts(
    path: str, options: dict, ignore_output: bool = False
) -> None:
    entrypoint_name = _get_required_option(options, "entrypoint")
    for entry in compat.importlib_metadata_get("console_scripts"):
        if entry.name == entrypoint_name:
            impl: Any = entry
            break
    else:
        raise util.CommandError(
            f"Could not find entrypoint console_scripts.{entrypoint_name}"
        )

    command = [
        sys.executable,
        "-c",
        f"import {impl.module}; {impl.module}.{impl.attr}()",
    ]
    _run_hook(path, options, ignore_output, command)


@register("exec")
def exec_(path: str, options: dict, ignore_output: bool = False) -> None:
    executable = _get_required_option(options, "executable")
    _run_hook(path, options, ignore_output, command=[executable])


@register("module")
def module(path: str, options: dict, ignore_output: bool = False) -> None:
    module_name = _get_required_option(options, "module")

    if importlib.util.find_spec(module_name) is None:
        raise util.CommandError(f"Could not find module {module_name}")

    command = [sys.executable, "-m", module_name]
    _run_hook(path, options, ignore_output, command)
