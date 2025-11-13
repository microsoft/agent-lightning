from __future__ import annotations

from argparse import ArgumentParser
from argparse import Namespace
from configparser import ConfigParser
import inspect
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import cast
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import TextIO
from typing import Union

from typing_extensions import TypedDict

from . import __version__
from . import command
from . import util
from .util import compat
from .util.pyfiles import _preserving_path_as_str


class Config:
    r"""Represent an Alembic configuration.

    Within an ``env.py`` script, this is available
    via the :attr:`.EnvironmentContext.config` attribute,
    which in turn is available at ``alembic.context``::

        from alembic import context

        some_param = context.config.get_main_option("my option")

    When invoking Alembic programmatically, a new
    :class:`.Config` can be created by passing
    the name of an .ini file to the constructor::

        from alembic.config import Config
        alembic_cfg = Config("/path/to/yourapp/alembic.ini")

    With a :class:`.Config` object, you can then
    run Alembic commands programmatically using the directives
    in :mod:`alembic.command`.

    The :class:`.Config` object can also be constructed without
    a filename.   Values can be set programmatically, and
    new sections will be created as needed::

        from alembic.config import Config
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", "myapp:migrations")
        alembic_cfg.set_main_option("sqlalchemy.url", "postgresql://foo/bar")
        alembic_cfg.set_section_option("mysection", "foo", "bar")

    .. warning::

       When using programmatic configuration, make sure the
       ``env.py`` file in use is compatible with the target configuration;
       including that the call to Python ``logging.fileConfig()`` is
       omitted if the programmatic configuration doesn't actually include
       logging directives.

    For passing non-string values to environments, such as connections and
    engines, use the :attr:`.Config.attributes` dictionary::

        with engine.begin() as connection:
            alembic_cfg.attributes['connection'] = connection
            command.upgrade(alembic_cfg, "head")

    :param file\_: name of the .ini file to open if an ``alembic.ini`` is
     to be used.    This should refer to the ``alembic.ini`` file, either as
     a filename or a full path to the file.  This filename if passed must refer
     to an **ini file in ConfigParser format** only.

    :param toml\_file: name of the pyproject.toml file to open if a
     ``pyproject.toml`` file is to be used.  This should refer to the
     ``pyproject.toml`` file, either as a filename or a full path to the file.
     This file must be in toml format. Both :paramref:`.Config.file\_` and
     :paramref:`.Config.toml\_file` may be passed simultaneously, or
     exclusively.

     .. versionadded:: 1.16.0

    :param ini_section: name of the main Alembic section within the
     .ini file
    :param output_buffer: optional file-like input buffer which
     will be passed to the :class:`.MigrationContext` - used to redirect
     the output of "offline generation" when using Alembic programmatically.
    :param stdout: buffer where the "print" output of commands will be sent.
     Defaults to ``sys.stdout``.

    :param config_args: A dictionary of keys and values that will be used
     for substitution in the alembic config file, as well as the pyproject.toml
     file, depending on which / both are used.  The dictionary as given is
     **copied** to two new, independent dictionaries, stored locally under the
     attributes ``.config_args`` and ``.toml_args``.   Both of these
     dictionaries will also be populated with the replacement variable
     ``%(here)s``, which refers to the location of the .ini and/or .toml file
     as appropriate.

    :param attributes: optional dictionary of arbitrary Python keys/values,
     which will be populated into the :attr:`.Config.attributes` dictionary.

     .. seealso::

        :ref:`connection_sharing`

    """

    def __init__(
        self,
        file_: Union[str, os.PathLike[str], None] = None,
        toml_file: Union[str, os.PathLike[str], None] = None,
        ini_section: str = "alembic",
        output_buffer: Optional[TextIO] = None,
        stdout: TextIO = sys.stdout,
        cmd_opts: Optional[Namespace] = None,
        config_args: Mapping[str, Any] = util.immutabledict(),
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a new :class:`.Config`"""
        self.config_file_name = (
            _preserving_path_as_str(file_) if file_ else None
        )
        self.toml_file_name = (
            _preserving_path_as_str(toml_file) if toml_file else None
        )
        self.config_ini_section = ini_section
        self.output_buffer = output_buffer
        self.stdout = stdout
        self.cmd_opts = cmd_opts
        self.config_args = dict(config_args)
        self.toml_args = dict(config_args)
        if attributes:
            self.attributes.update(attributes)

    cmd_opts: Optional[Namespace] = None
    """The command-line options passed to the ``alembic`` script.

    Within an ``env.py`` script this can be accessed via the
    :attr:`.EnvironmentContext.config` attribute.

    .. seealso::

        :meth:`.EnvironmentContext.get_x_argument`

    """

    config_file_name: Optional[str] = None
    """Filesystem path to the .ini file in use."""

    toml_file_name: Optional[str] = None
    """Filesystem path to the pyproject.toml file in use.

    .. versionadded:: 1.16.0

    """

    @property
    def _config_file_path(self) -> Optional[Path]:
        if self.config_file_name is None:
            return None
        return Path(self.config_file_name)

    @property
    def _toml_file_path(self) -> Optional[Path]:
        if self.toml_file_name is None:
            return None
        return Path(self.toml_file_name)

    config_ini_section: str = None  # type:ignore[assignment]
    """Name of the config file section to read basic configuration
    from.  Defaults to ``alembic``, that is the ``[alembic]`` section
    of the .ini file.  This value is modified using the ``-n/--name``
    option to the Alembic runner.

    """

    @util.memoized_property
    def attributes(self) -> Dict[str, Any]:
        """A Python dictionary for storage of additional state.


        This is a utility dictionary which can include not just strings but
        engines, connections, schema objects, or anything else.
        Use this to pass objects into an env.py script, such as passing
        a :class:`sqlalchemy.engine.base.Connection` when calling
        commands from :mod:`alembic.command` programmatically.

        .. seealso::

            :ref:`connection_sharing`

            :paramref:`.Config.attributes`

        """
        return {}

    def print_stdout(self, text: str, *arg: Any) -> None:
        """Render a message to standard out.

        When :meth:`.Config.print_stdout` is called with additional args
        those arguments will formatted against the provided text,
        otherwise we simply output the provided text verbatim.

        This is a no-op when the``quiet`` messaging option is enabled.

        e.g.::

            >>> config.print_stdout('Some text %s', 'arg')
            Some Text arg

        """

        if arg:
            output = str(text) % arg
        else:
            output = str(text)

        util.write_outstream(self.stdout, output, "\n", **self.messaging_opts)

    @util.memoized_property
    def file_config(self) -> ConfigParser:
        """Return the underlying ``ConfigParser`` object.

        Dir*-ect access to the .ini file is available here,
        though the :meth:`.Config.get_section` and
        :meth:`.Config.get_main_option`
        methods provide a possibly simpler interface.

        """

        if self._config_file_path:
            here = self._config_file_path.absolute().parent
        else:
            here = Path()
        self.config_args["here"] = here.as_posix()
        file_config = ConfigParser(self.config_args)
        if self._config_file_path:
            compat.read_config_parser(file_config, [self._config_file_path])
        else:
            file_config.add_section(self.config_ini_section)
        return file_config

    @util.memoized_property
    def toml_alembic_config(self) -> Mapping[str, Any]:
        """Return a dictionary of the [tool.alembic] section from
        pyproject.toml"""

        if self._toml_file_path and self._toml_file_path.exists():

            here = self._toml_file_path.absolute().parent
            self.toml_args["here"] = here.as_posix()

            with open(self._toml_file_path, "rb") as f:
                toml_data = compat.tomllib.load(f)
                data = toml_data.get("tool", {}).get("alembic", {})
                if not isinstance(data, dict):
                    raise util.CommandError("Incorrect TOML format")
                return data

        else:
            return {}

    def get_template_directory(self) -> str:
        """Return the directory where Alembic setup templates are found.

        This method is used by the alembic ``init`` and ``list_templates``
        commands.

        """
        import alembic

        package_dir = Path(alembic.__file__).absolute().parent
        return str(package_dir / "templates")

    def _get_template_path(self) -> Path:
        """Return the directory where Alembic setup templates are found.

        This method is used by the alembic ``init`` and ``list_templates``
        commands.

        .. versionadded:: 1.16.0

        """
        return Path(self.get_template_directory())

    @overload
    def get_section(
        self, name: str, default: None = ...
    ) -> Optional[Dict[str, str]]: ...

    # "default" here could also be a TypeVar
    # _MT = TypeVar("_MT", bound=Mapping[str, str]),
    # however mypy wasn't handling that correctly (pyright was)
    @overload
    def get_section(
        self, name: str, default: Dict[str, str]
    ) -> Dict[str, str]: ...

    @overload
    def get_section(
        self, name: str, default: Mapping[str, str]
    ) -> Union[Dict[str, str], Mapping[str, str]]: ...

    def get_section(
        self, name: str, default: Optional[Mapping[str, str]] = None
    ) -> Optional[Mapping[str, str]]:
        """Return all the configuration options from a given .ini file section
        as a dictionary.

        If the given section does not exist, the value of ``default``
        is returned, which is expected to be a dictionary or other mapping.

        """
        if not self.file_config.has_section(name):
            return default

        return dict(self.file_config.items(name))

    def set_main_option(self, name: str, value: str) -> None:
        """Set an option programmatically within the 'main' section.

        This overrides whatever was in the .ini file.

        :param name: name of the value

        :param value: the value.  Note that this value is passed to
         ``ConfigParser.set``, which supports variable interpolation using
         pyformat (e.g. ``%(some_value)s``).   A raw percent sign not part of
         an interpolation symbol must therefore be escaped, e.g. ``%%``.
         The given value may refer to another value already in the file
         using the interpolation format.

        """
        self.set_section_option(self.config_ini_section, name, value)

    def remove_main_option(self, name: str) -> None:
        self.file_config.remove_option(self.config_ini_section, name)

    def set_section_option(self, section: str, name: str, value: str) -> None:
        """Set an option programmatically within the given section.

        The section is created if it doesn't exist already.
        The value here will override whatever was in the .ini
        file.

        Does **NOT** consume from the pyproject.toml file.

        .. seealso::

            :meth:`.Config.get_alembic_option` - includes pyproject support

        :param section: name of the section

        :param name: name of the value

        :param value: the value.  Note that this value is passed to
         ``ConfigParser.set``, which supports variable interpolation using
         pyformat (e.g. ``%(some_value)s``).   A raw percent sign not part of
         an interpolation symbol must therefore be escaped, e.g. ``%%``.
         The given value may refer to another value already in the file
         using the interpolation format.

        """

        if not self.file_config.has_section(section):
            self.file_config.add_section(section)
        self.file_config.set(section, name, value)

    def get_section_option(
        self, section: str, name: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Return an option from the given section of the .ini file."""
        if not self.file_config.has_section(section):
            raise util.CommandError(
                "No config file %r found, or file has no "
                "'[%s]' section" % (self.config_file_name, section)
            )
        if self.file_config.has_option(section, name):
            return self.file_config.get(section, name)
        else:
            return default

    @overload
    def get_main_option(self, name: str, default: str) -> str: ...

    @overload
    def get_main_option(
        self, name: str, default: Optional[str] = None
    ) -> Optional[str]: ...

    def get_main_option(
        self, name: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Return an option from the 'main' section of the .ini file.

        This defaults to being a key from the ``[alembic]``
        section, unless the ``-n/--name`` flag were used to
        indicate a different section.

        Does **NOT** consume from the pyproject.toml file.

        .. seealso::

            :meth:`.Config.get_alembic_option` - includes pyproject support

        """
        return self.get_section_option(self.config_ini_section, name, default)

    @overload
    def get_alembic_option(self, name: str, default: str) -> str: ...

    @overload
    def get_alembic_option(
        self, name: str, default: Optional[str] = None
    ) -> Optional[str]: ...

    def get_alembic_option(
        self, name: str, default: Optional[str] = None
    ) -> Union[
        None, str, list[str], dict[str, str], list[dict[str, str]], int
    ]:
        """Return an option from the "[alembic]" or "[tool.alembic]" section
        of the configparser-parsed .ini file (e.g. ``alembic.ini``) or
        toml-parsed ``pyproject.toml`` file.

        The value returned is expected to be None, string, list of strings,
        or dictionary of strings.   Within each type of string value, the
        ``%(here)s`` token is substituted out with the absolute path of the
        ``pyproject.toml`` file, as are other tokens which are extracted from
        the :paramref:`.Config.config_args` dictionary.

        Searches always prioritize the configparser namespace first, before
        searching in the toml namespace.

        If Alembic was run using the ``-n/--name`` flag to indicate an
        alternate main section name, this is taken into account **only** for
        the configparser-parsed .ini file.  The section name in toml is always
        ``[tool.alembic]``.


        .. versionadded:: 1.16.0

        """

        if self.file_config.has_option(self.config_ini_section, name):
            return self.file_config.get(self.config_ini_section, name)
        else:
            return self._get_toml_config_value(name, default=default)

    def get_alembic_boolean_option(self, name: str) -> bool:
        if self.file_config.has_option(self.config_ini_section, name):
            return (
                self.file_config.get(self.config_ini_section, name) == "true"
            )
        else:
            value = self.toml_alembic_config.get(name, False)
            if not isinstance(value, bool):
                raise util.CommandError(
                    f"boolean value expected for TOML parameter {name!r}"
                )
            return value

    def _get_toml_config_value(
        self, name: str, default: Optional[Any] = None
    ) -> Union[
        None, str, list[str], dict[str, str], list[dict[str, str]], int
    ]:
        USE_DEFAULT = object()
        value: Union[None, str, list[str], dict[str, str], int] = (
            self.toml_alembic_config.get(name, USE_DEFAULT)
        )
        if value is USE_DEFAULT:
            return default
        if value is not None:
            if isinstance(value, str):
                value = value % (self.toml_args)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    value = [
                        {k: v % (self.toml_args) for k, v in dv.items()}
                        for dv in value
                    ]
                else:
                    value = cast(
                        "list[str]", [v % (self.toml_args) for v in value]
                    )
            elif isinstance(value, dict):
                value = cast(
                    "dict[str, str]",
                    {k: v % (self.toml_args) for k, v in value.items()},
                )
            elif isinstance(value, int):
                return value
            else:
                raise util.CommandError(
                    f"unsupported TOML value type for key: {name!r}"
                )
        return value

    @util.memoized_property
    def messaging_opts(self) -> MessagingOptions:
        """The messaging options."""
        return cast(
            MessagingOptions,
            util.immutabledict(
                {"quiet": getattr(self.cmd_opts, "quiet", False)}
            ),
        )

    def _get_file_separator_char(self, *names: str) -> Optional[str]:
        for name in names:
            separator = self.get_main_option(name)
            if separator is not None:
                break
        else:
            return None

        split_on_path = {
            "space": " ",
            "newline": "\n",
            "os": os.pathsep,
            ":": ":",
            ";": ";",
        }

        try:
            sep = split_on_path[separator]
        except KeyError as ke:
            raise ValueError(
                "'%s' is not a valid value for %s; "
                "expected 'space', 'newline', 'os', ':', ';'"
                % (separator, name)
            ) from ke
        else:
            if name == "version_path_separator":
                util.warn_deprecated(
                    "The version_path_separator configuration parameter "
                    "is deprecated; please use path_separator"
                )
            return sep

    def get_version_locations_list(self) -> Optional[list[str]]:

        version_locations_str = self.file_config.get(
            self.config_ini_section, "version_locations", fallback=None
        )

        if version_locations_str:
            split_char = self._get_file_separator_char(
                "path_separator", "version_path_separator"
            )

            if split_char is None:

                # legacy behaviour for backwards compatibility
                util.warn_deprecated(
                    "No path_separator found in configuration; "
                    "falling back to legacy splitting on spaces/commas "
                    "for version_locations.  Consider adding "
                    "path_separator=os to Alembic config."
                )

                _split_on_space_comma = re.compile(r", *|(?: +)")
                return _split_on_space_comma.split(version_locations_str)
            else:
                return [
                    x.strip()
                    for x in version_locations_str.split(split_char)
                    if x
                ]
        else:
            return cast(
                "list[str]",
                self._get_toml_config_value("version_locations", None),
            )

    def get_prepend_sys_paths_list(self) -> Optional[list[str]]:
        prepend_sys_path_str = self.file_config.get(
            self.config_ini_section, "prepend_sys_path", fallback=None
        )

        if prepend_sys_path_str:
            split_char = self._get_file_separator_char("path_separator")

            if split_char is None:

                # legacy behaviour for backwards compatibility
                util.warn_deprecated(
                    "No path_separator found in configuration; "
                    "falling back to legacy splitting on spaces, commas, "
                    "and colons for prepend_sys_path.  Consider adding "
                    "path_separator=os to Alembic config."
                )

                _split_on_space_comma_colon = re.compile(r", *|(?: +)|\:")
                return _split_on_space_comma_colon.split(prepend_sys_path_str)
            else:
                return [
                    x.strip()
                    for x in prepend_sys_path_str.split(split_char)
                    if x
                ]
        else:
            return cast(
                "list[str]",
                self._get_toml_config_value("prepend_sys_path", None),
            )

    def get_hooks_list(self) -> list[PostWriteHookConfig]:

        hooks: list[PostWriteHookConfig] = []

        if not self.file_config.has_section("post_write_hooks"):
            toml_hook_config = cast(
                "list[dict[str, str]]",
                self._get_toml_config_value("post_write_hooks", []),
            )
            for cfg in toml_hook_config:
                opts = dict(cfg)
                opts["_hook_name"] = opts.pop("name")
                hooks.append(opts)

        else:
            _split_on_space_comma = re.compile(r", *|(?: +)")
            ini_hook_config = self.get_section("post_write_hooks", {})
            names = _split_on_space_comma.split(
                ini_hook_config.get("hooks", "")
            )

            for name in names:
                if not name:
                    continue
                opts = {
                    key[len(name) + 1 :]: ini_hook_config[key]
                    for key in ini_hook_config
                    if key.startswith(name + ".")
                }

                opts["_hook_name"] = name
                hooks.append(opts)

        return hooks


PostWriteHookConfig = Mapping[str, str]


class MessagingOptions(TypedDict, total=False):
    quiet: bool


class CommandFunction(Protocol):
    """A function that may be registered in the CLI as an alembic command.
    It must be a named function and it must accept a :class:`.Config` object
    as the first argument.

    .. versionadded:: 1.15.3

    """

    __name__: str

    def __call__(self, config: Config, *args: Any, **kwargs: Any) -> Any: ...


class CommandLine:
    """Provides the command line interface to Alembic."""

    def __init__(self, prog: Optional[str] = None) -> None:
        self._generate_args(prog)

    _KWARGS_OPTS = {
        "template": (
            "-t",
            "--template",
            dict(
                default="generic",
                type=str,
                help="Setup template for use with 'init'",
            ),
        ),
        "message": (
            "-m",
            "--message",
            dict(type=str, help="Message string to use with 'revision'"),
        ),
        "sql": (
            "--sql",
            dict(
                action="store_true",
                help="Don't emit SQL to database - dump to "
                "standard output/file instead. See docs on "
                "offline mode.",
            ),
        ),
        "tag": (
            "--tag",
            dict(
                type=str,
                help="Arbitrary 'tag' name - can be used by "
                "custom env.py scripts.",
            ),
        ),
        "head": (
            "--head",
            dict(
                type=str,
                help="Specify head revision or <branchname>@head "
                "to base new revision on.",
            ),
        ),
        "splice": (
            "--splice",
            dict(
                action="store_true",
                help="Allow a non-head revision as the 'head' to splice onto",
            ),
        ),
        "depends_on": (
            "--depends-on",
            dict(
                action="append",
                help="Specify one or more revision identifiers "
                "which this revision should depend on.",
            ),
        ),
        "rev_id": (
            "--rev-id",
            dict(
                type=str,
                help="Specify a hardcoded revision id instead of "
                "generating one",
            ),
        ),
        "version_path": (
            "--version-path",
            dict(
                type=str,
                help="Specify specific path from config for version file",
            ),
        ),
        "branch_label": (
            "--branch-label",
            dict(
                type=str,
                help="Specify a branch label to apply to the new revision",
            ),
        ),
        "verbose": (
            "-v",
            "--verbose",
            dict(action="store_true", help="Use more verbose output"),
        ),
        "resolve_dependencies": (
            "--resolve-dependencies",
            dict(
                action="store_true",
                help="Treat dependency versions as down revisions",
            ),
        ),
        "autogenerate": (
            "--autogenerate",
            dict(
                action="store_true",
                help="Populate revision script with candidate "
                "migration operations, based on comparison "
                "of database to model.",
            ),
        ),
        "rev_range": (
            "-r",
            "--rev-range",
            dict(
                action="store",
                help="Specify a revision range; format is [start]:[end]",
            ),
        ),
        "indicate_current": (
            "-i",
            "--indicate-current",
            dict(
                action="store_true",
                help="Indicate the current revision",
            ),
        ),
        "purge": (
            "--purge",
            dict(
                action="store_true",
                help="Unconditionally erase the version table before stamping",
            ),
        ),
        "package": (
            "--package",
            dict(
                action="store_true",
                help="Write empty __init__.py files to the "
                "environment and version locations",
            ),
        ),
    }
    _POSITIONAL_OPTS = {
        "directory": dict(help="location of scripts directory"),
        "revision": dict(
            help="revision identifier",
        ),
        "revisions": dict(
            nargs="+",
            help="one or more revisions, or 'heads' for all heads",
        ),
    }
    _POSITIONAL_TRANSLATIONS: dict[Any, dict[str, str]] = {
        command.stamp: {"revision": "revisions"}
    }

    def _generate_args(self, prog: Optional[str]) -> None:
        parser = ArgumentParser(prog=prog)

        parser.add_argument(
            "--version", action="version", version="%%(prog)s %s" % __version__
        )
        parser.add_argument(
            "-c",
            "--config",
            action="append",
            help="Alternate config file; defaults to value of "
            'ALEMBIC_CONFIG environment variable, or "alembic.ini". '
            "May also refer to pyproject.toml file.  May be specified twice "
            "to reference both files separately",
        )
        parser.add_argument(
            "-n",
            "--name",
            type=str,
            default="alembic",
            help="Name of section in .ini file to use for Alembic config "
            "(only applies to configparser config, not toml)",
        )
        parser.add_argument(
            "-x",
            action="append",
            help="Additional arguments consumed by "
            "custom env.py scripts, e.g. -x "
            "setting1=somesetting -x setting2=somesetting",
        )
        parser.add_argument(
            "--raiseerr",
            action="store_true",
            help="Raise a full stack trace on error",
        )
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Do not log to std output.",
        )

        self.subparsers = parser.add_subparsers()
        alembic_commands = (
            cast(CommandFunction, fn)
            for fn in (getattr(command, name) for name in dir(command))
            if (
                inspect.isfunction(fn)
                and fn.__name__[0] != "_"
                and fn.__module__ == "alembic.command"
            )
        )

        for fn in alembic_commands:
            self.register_command(fn)

        self.parser = parser

    def register_command(self, fn: CommandFunction) -> None:
        """Registers a function as a CLI subcommand. The subcommand name
        matches the function name, the arguments are extracted from the
        signature and the help text is read from the docstring.

        .. versionadded:: 1.15.3

        .. seealso::

            :ref:`custom_commandline`
        """

        positional, kwarg, help_text = self._inspect_function(fn)

        subparser = self.subparsers.add_parser(fn.__name__, help=help_text)
        subparser.set_defaults(cmd=(fn, positional, kwarg))

        for arg in kwarg:
            if arg in self._KWARGS_OPTS:
                kwarg_opt = self._KWARGS_OPTS[arg]
                args, opts = kwarg_opt[0:-1], kwarg_opt[-1]
                subparser.add_argument(*args, **opts)  # type:ignore

        for arg in positional:
            opts = self._POSITIONAL_OPTS.get(arg, {})
            subparser.add_argument(arg, **opts)  # type:ignore

    def _inspect_function(self, fn: CommandFunction) -> tuple[Any, Any, str]:
        spec = compat.inspect_getfullargspec(fn)
        if spec[3] is not None:
            positional = spec[0][1 : -len(spec[3])]
            kwarg = spec[0][-len(spec[3]) :]
        else:
            positional = spec[0][1:]
            kwarg = []

        if fn in self._POSITIONAL_TRANSLATIONS:
            positional = [
                self._POSITIONAL_TRANSLATIONS[fn].get(name, name)
                for name in positional
            ]

        # parse first line(s) of helptext without a line break
        help_ = fn.__doc__
        if help_:
            help_lines = []
            for line in help_.split("\n"):
                if not line.strip():
                    break
                else:
                    help_lines.append(line.strip())
        else:
            help_lines = []

        help_text = " ".join(help_lines)

        return positional, kwarg, help_text

    def run_cmd(self, config: Config, options: Namespace) -> None:
        fn, positional, kwarg = options.cmd

        try:
            fn(
                config,
                *[getattr(options, k, None) for k in positional],
                **{k: getattr(options, k, None) for k in kwarg},
            )
        except util.CommandError as e:
            if options.raiseerr:
                raise
            else:
                util.err(str(e), **config.messaging_opts)

    def _inis_from_config(self, options: Namespace) -> tuple[str, str]:
        names = options.config

        alembic_config_env = os.environ.get("ALEMBIC_CONFIG")
        if (
            alembic_config_env
            and os.path.basename(alembic_config_env) == "pyproject.toml"
        ):
            default_pyproject_toml = alembic_config_env
            default_alembic_config = "alembic.ini"
        elif alembic_config_env:
            default_pyproject_toml = "pyproject.toml"
            default_alembic_config = alembic_config_env
        else:
            default_alembic_config = "alembic.ini"
            default_pyproject_toml = "pyproject.toml"

        if not names:
            return default_pyproject_toml, default_alembic_config

        toml = ini = None

        for name in names:
            if os.path.basename(name) == "pyproject.toml":
                if toml is not None:
                    raise util.CommandError(
                        "pyproject.toml indicated more than once"
                    )
                toml = name
            else:
                if ini is not None:
                    raise util.CommandError(
                        "only one ini file may be indicated"
                    )
                ini = name

        return toml if toml else default_pyproject_toml, (
            ini if ini else default_alembic_config
        )

    def main(self, argv: Optional[Sequence[str]] = None) -> None:
        """Executes the command line with the provided arguments."""
        options = self.parser.parse_args(argv)
        if not hasattr(options, "cmd"):
            # see http://bugs.python.org/issue9253, argparse
            # behavior changed incompatibly in py3.3
            self.parser.error("too few arguments")
        else:
            toml, ini = self._inis_from_config(options)
            cfg = Config(
                file_=ini,
                toml_file=toml,
                ini_section=options.name,
                cmd_opts=options,
            )
            self.run_cmd(cfg, options)


def main(
    argv: Optional[Sequence[str]] = None,
    prog: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """The console runner function for Alembic."""

    CommandLine(prog=prog).main(argv=argv)


if __name__ == "__main__":
    main()
