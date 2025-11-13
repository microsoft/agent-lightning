from __future__ import annotations

import atexit
from contextlib import ExitStack
import importlib
import importlib.machinery
import importlib.util
import os
import pathlib
import re
import tempfile
from types import ModuleType
from typing import Any
from typing import Optional
from typing import Union

from mako import exceptions
from mako.template import Template

from . import compat
from .exc import CommandError


def template_to_file(
    template_file: Union[str, os.PathLike[str]],
    dest: Union[str, os.PathLike[str]],
    output_encoding: str,
    *,
    append_with_newlines: bool = False,
    **kw: Any,
) -> None:
    template = Template(filename=_preserving_path_as_str(template_file))
    try:
        output = template.render_unicode(**kw).encode(output_encoding)
    except:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as ntf:
            ntf.write(
                exceptions.text_error_template()
                .render_unicode()
                .encode(output_encoding)
            )
            fname = ntf.name
        raise CommandError(
            "Template rendering failed; see %s for a "
            "template-oriented traceback." % fname
        )
    else:
        with open(dest, "ab" if append_with_newlines else "wb") as f:
            if append_with_newlines:
                f.write("\n\n".encode(output_encoding))
            f.write(output)


def coerce_resource_to_filename(fname_or_resource: str) -> pathlib.Path:
    """Interpret a filename as either a filesystem location or as a package
    resource.

    Names that are non absolute paths and contain a colon
    are interpreted as resources and coerced to a file location.

    """
    # TODO: there seem to be zero tests for the package resource codepath
    if not os.path.isabs(fname_or_resource) and ":" in fname_or_resource:
        tokens = fname_or_resource.split(":")

        # from https://importlib-resources.readthedocs.io/en/latest/migration.html#pkg-resources-resource-filename  # noqa E501

        file_manager = ExitStack()
        atexit.register(file_manager.close)

        ref = compat.importlib_resources.files(tokens[0])
        for tok in tokens[1:]:
            ref = ref / tok
        fname_or_resource = file_manager.enter_context(  # type: ignore[assignment]  # noqa: E501
            compat.importlib_resources.as_file(ref)
        )
    return pathlib.Path(fname_or_resource)


def pyc_file_from_path(
    path: Union[str, os.PathLike[str]],
) -> Optional[pathlib.Path]:
    """Given a python source path, locate the .pyc."""

    pathpath = pathlib.Path(path)
    candidate = pathlib.Path(
        importlib.util.cache_from_source(pathpath.as_posix())
    )
    if candidate.exists():
        return candidate

    # even for pep3147, fall back to the old way of finding .pyc files,
    # to support sourceless operation
    ext = pathpath.suffix
    for ext in importlib.machinery.BYTECODE_SUFFIXES:
        if pathpath.with_suffix(ext).exists():
            return pathpath.with_suffix(ext)
    else:
        return None


def load_python_file(
    dir_: Union[str, os.PathLike[str]], filename: Union[str, os.PathLike[str]]
) -> ModuleType:
    """Load a file from the given path as a Python module."""

    dir_ = pathlib.Path(dir_)
    filename_as_path = pathlib.Path(filename)
    filename = filename_as_path.name

    module_id = re.sub(r"\W", "_", filename)
    path = dir_ / filename
    ext = path.suffix
    if ext == ".py":
        if path.exists():
            module = load_module_py(module_id, path)
        else:
            pyc_path = pyc_file_from_path(path)
            if pyc_path is None:
                raise ImportError("Can't find Python file %s" % path)
            else:
                module = load_module_py(module_id, pyc_path)
    elif ext in (".pyc", ".pyo"):
        module = load_module_py(module_id, path)
    else:
        assert False
    return module


def load_module_py(
    module_id: str, path: Union[str, os.PathLike[str]]
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_id, path)
    assert spec
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def _preserving_path_as_str(path: Union[str, os.PathLike[str]]) -> str:
    """receive str/pathlike and return a string.

    Does not convert an incoming string path to a Path first, to help with
    unit tests that are doing string path round trips without OS-specific
    processing if not necessary.

    """
    if isinstance(path, str):
        return path
    elif isinstance(path, pathlib.PurePath):
        return str(path)
    else:
        return str(pathlib.Path(path))
