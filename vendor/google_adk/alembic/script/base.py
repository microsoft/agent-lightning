from __future__ import annotations

from contextlib import contextmanager
import datetime
import os
from pathlib import Path
import re
import shutil
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from . import revision
from . import write_hooks
from .. import util
from ..runtime import migration
from ..util import compat
from ..util import not_none
from ..util.pyfiles import _preserving_path_as_str

if TYPE_CHECKING:
    from .revision import _GetRevArg
    from .revision import _RevIdType
    from .revision import Revision
    from ..config import Config
    from ..config import MessagingOptions
    from ..config import PostWriteHookConfig
    from ..runtime.migration import RevisionStep
    from ..runtime.migration import StampStep

try:
    if compat.py39:
        from zoneinfo import ZoneInfo
        from zoneinfo import ZoneInfoNotFoundError
    else:
        from backports.zoneinfo import ZoneInfo  # type: ignore[import-not-found,no-redef] # noqa: E501
        from backports.zoneinfo import ZoneInfoNotFoundError  # type: ignore[no-redef] # noqa: E501
except ImportError:
    ZoneInfo = None  # type: ignore[assignment, misc]

_sourceless_rev_file = re.compile(r"(?!\.\#|__init__)(.*\.py)(c|o)?$")
_only_source_rev_file = re.compile(r"(?!\.\#|__init__)(.*\.py)$")
_legacy_rev = re.compile(r"([a-f0-9]+)\.py$")
_slug_re = re.compile(r"\w+")
_default_file_template = "%(rev)s_%(slug)s"


class ScriptDirectory:
    """Provides operations upon an Alembic script directory.

    This object is useful to get information as to current revisions,
    most notably being able to get at the "head" revision, for schemes
    that want to test if the current revision in the database is the most
    recent::

        from alembic.script import ScriptDirectory
        from alembic.config import Config
        config = Config()
        config.set_main_option("script_location", "myapp:migrations")
        script = ScriptDirectory.from_config(config)

        head_revision = script.get_current_head()



    """

    def __init__(
        self,
        dir: Union[str, os.PathLike[str]],  # noqa: A002
        file_template: str = _default_file_template,
        truncate_slug_length: Optional[int] = 40,
        version_locations: Optional[
            Sequence[Union[str, os.PathLike[str]]]
        ] = None,
        sourceless: bool = False,
        output_encoding: str = "utf-8",
        timezone: Optional[str] = None,
        hooks: list[PostWriteHookConfig] = [],
        recursive_version_locations: bool = False,
        messaging_opts: MessagingOptions = cast(
            "MessagingOptions", util.EMPTY_DICT
        ),
    ) -> None:
        self.dir = _preserving_path_as_str(dir)
        self.version_locations = [
            _preserving_path_as_str(p) for p in version_locations or ()
        ]
        self.file_template = file_template
        self.truncate_slug_length = truncate_slug_length or 40
        self.sourceless = sourceless
        self.output_encoding = output_encoding
        self.revision_map = revision.RevisionMap(self._load_revisions)
        self.timezone = timezone
        self.hooks = hooks
        self.recursive_version_locations = recursive_version_locations
        self.messaging_opts = messaging_opts

        if not os.access(dir, os.F_OK):
            raise util.CommandError(
                f"Path doesn't exist: {dir}.  Please use "
                "the 'init' command to create a new "
                "scripts folder."
            )

    @property
    def versions(self) -> str:
        """return a single version location based on the sole path passed
        within version_locations.

        If multiple version locations are configured, an error is raised.


        """
        return str(self._singular_version_location)

    @util.memoized_property
    def _singular_version_location(self) -> Path:
        loc = self._version_locations
        if len(loc) > 1:
            raise util.CommandError("Multiple version_locations present")
        else:
            return loc[0]

    @util.memoized_property
    def _version_locations(self) -> Sequence[Path]:
        if self.version_locations:
            return [
                util.coerce_resource_to_filename(location).absolute()
                for location in self.version_locations
            ]
        else:
            return [Path(self.dir, "versions").absolute()]

    def _load_revisions(self) -> Iterator[Script]:
        paths = [vers for vers in self._version_locations if vers.exists()]

        dupes = set()
        for vers in paths:
            for file_path in Script._list_py_dir(self, vers):
                real_path = file_path.resolve()
                if real_path in dupes:
                    util.warn(
                        f"File {real_path} loaded twice! ignoring. "
                        "Please ensure version_locations is unique."
                    )
                    continue
                dupes.add(real_path)

                script = Script._from_path(self, real_path)
                if script is None:
                    continue
                yield script

    @classmethod
    def from_config(cls, config: Config) -> ScriptDirectory:
        """Produce a new :class:`.ScriptDirectory` given a :class:`.Config`
        instance.

        The :class:`.Config` need only have the ``script_location`` key
        present.

        """
        script_location = config.get_alembic_option("script_location")
        if script_location is None:
            raise util.CommandError(
                "No 'script_location' key found in configuration."
            )
        truncate_slug_length: Optional[int]
        tsl = config.get_alembic_option("truncate_slug_length")
        if tsl is not None:
            truncate_slug_length = int(tsl)
        else:
            truncate_slug_length = None

        prepend_sys_path = config.get_prepend_sys_paths_list()
        if prepend_sys_path:
            sys.path[:0] = prepend_sys_path

        rvl = config.get_alembic_boolean_option("recursive_version_locations")
        return ScriptDirectory(
            util.coerce_resource_to_filename(script_location),
            file_template=config.get_alembic_option(
                "file_template", _default_file_template
            ),
            truncate_slug_length=truncate_slug_length,
            sourceless=config.get_alembic_boolean_option("sourceless"),
            output_encoding=config.get_alembic_option(
                "output_encoding", "utf-8"
            ),
            version_locations=config.get_version_locations_list(),
            timezone=config.get_alembic_option("timezone"),
            hooks=config.get_hooks_list(),
            recursive_version_locations=rvl,
            messaging_opts=config.messaging_opts,
        )

    @contextmanager
    def _catch_revision_errors(
        self,
        ancestor: Optional[str] = None,
        multiple_heads: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> Iterator[None]:
        try:
            yield
        except revision.RangeNotAncestorError as rna:
            if start is None:
                start = cast(Any, rna.lower)
            if end is None:
                end = cast(Any, rna.upper)
            if not ancestor:
                ancestor = (
                    "Requested range %(start)s:%(end)s does not refer to "
                    "ancestor/descendant revisions along the same branch"
                )
            ancestor = ancestor % {"start": start, "end": end}
            raise util.CommandError(ancestor) from rna
        except revision.MultipleHeads as mh:
            if not multiple_heads:
                multiple_heads = (
                    "Multiple head revisions are present for given "
                    "argument '%(head_arg)s'; please "
                    "specify a specific target revision, "
                    "'<branchname>@%(head_arg)s' to "
                    "narrow to a specific head, or 'heads' for all heads"
                )
            multiple_heads = multiple_heads % {
                "head_arg": end or mh.argument,
                "heads": util.format_as_comma(mh.heads),
            }
            raise util.CommandError(multiple_heads) from mh
        except revision.ResolutionError as re:
            if resolution is None:
                resolution = "Can't locate revision identified by '%s'" % (
                    re.argument
                )
            raise util.CommandError(resolution) from re
        except revision.RevisionError as err:
            raise util.CommandError(err.args[0]) from err

    def walk_revisions(
        self, base: str = "base", head: str = "heads"
    ) -> Iterator[Script]:
        """Iterate through all revisions.

        :param base: the base revision, or "base" to start from the
         empty revision.

        :param head: the head revision; defaults to "heads" to indicate
         all head revisions.  May also be "head" to indicate a single
         head revision.

        """
        with self._catch_revision_errors(start=base, end=head):
            for rev in self.revision_map.iterate_revisions(
                head, base, inclusive=True, assert_relative_length=False
            ):
                yield cast(Script, rev)

    def get_revisions(self, id_: _GetRevArg) -> Tuple[Script, ...]:
        """Return the :class:`.Script` instance with the given rev identifier,
        symbolic name, or sequence of identifiers.

        """
        with self._catch_revision_errors():
            return cast(
                Tuple[Script, ...],
                self.revision_map.get_revisions(id_),
            )

    def get_all_current(self, id_: Tuple[str, ...]) -> Set[Script]:
        with self._catch_revision_errors():
            return cast(Set[Script], self.revision_map._get_all_current(id_))

    def get_revision(self, id_: str) -> Script:
        """Return the :class:`.Script` instance with the given rev id.

        .. seealso::

            :meth:`.ScriptDirectory.get_revisions`

        """

        with self._catch_revision_errors():
            return cast(Script, self.revision_map.get_revision(id_))

    def as_revision_number(
        self, id_: Optional[str]
    ) -> Optional[Union[str, Tuple[str, ...]]]:
        """Convert a symbolic revision, i.e. 'head' or 'base', into
        an actual revision number."""

        with self._catch_revision_errors():
            rev, branch_name = self.revision_map._resolve_revision_number(id_)

        if not rev:
            # convert () to None
            return None
        elif id_ == "heads":
            return rev
        else:
            return rev[0]

    def iterate_revisions(
        self,
        upper: Union[str, Tuple[str, ...], None],
        lower: Union[str, Tuple[str, ...], None],
        **kw: Any,
    ) -> Iterator[Script]:
        """Iterate through script revisions, starting at the given
        upper revision identifier and ending at the lower.

        The traversal uses strictly the `down_revision`
        marker inside each migration script, so
        it is a requirement that upper >= lower,
        else you'll get nothing back.

        The iterator yields :class:`.Script` objects.

        .. seealso::

            :meth:`.RevisionMap.iterate_revisions`

        """
        return cast(
            Iterator[Script],
            self.revision_map.iterate_revisions(upper, lower, **kw),
        )

    def get_current_head(self) -> Optional[str]:
        """Return the current head revision.

        If the script directory has multiple heads
        due to branching, an error is raised;
        :meth:`.ScriptDirectory.get_heads` should be
        preferred.

        :return: a string revision number.

        .. seealso::

            :meth:`.ScriptDirectory.get_heads`

        """
        with self._catch_revision_errors(
            multiple_heads=(
                "The script directory has multiple heads (due to branching)."
                "Please use get_heads(), or merge the branches using "
                "alembic merge."
            )
        ):
            return self.revision_map.get_current_head()

    def get_heads(self) -> List[str]:
        """Return all "versioned head" revisions as strings.

        This is normally a list of length one,
        unless branches are present.  The
        :meth:`.ScriptDirectory.get_current_head()` method
        can be used normally when a script directory
        has only one head.

        :return: a tuple of string revision numbers.
        """
        return list(self.revision_map.heads)

    def get_base(self) -> Optional[str]:
        """Return the "base" revision as a string.

        This is the revision number of the script that
        has a ``down_revision`` of None.

        If the script directory has multiple bases, an error is raised;
        :meth:`.ScriptDirectory.get_bases` should be
        preferred.

        """
        bases = self.get_bases()
        if len(bases) > 1:
            raise util.CommandError(
                "The script directory has multiple bases. "
                "Please use get_bases()."
            )
        elif bases:
            return bases[0]
        else:
            return None

    def get_bases(self) -> List[str]:
        """return all "base" revisions as strings.

        This is the revision number of all scripts that
        have a ``down_revision`` of None.

        """
        return list(self.revision_map.bases)

    def _upgrade_revs(
        self, destination: str, current_rev: str
    ) -> List[RevisionStep]:
        with self._catch_revision_errors(
            ancestor="Destination %(end)s is not a valid upgrade "
            "target from current head(s)",
            end=destination,
        ):
            revs = self.iterate_revisions(
                destination, current_rev, implicit_base=True
            )
            return [
                migration.MigrationStep.upgrade_from_script(
                    self.revision_map, script
                )
                for script in reversed(list(revs))
            ]

    def _downgrade_revs(
        self, destination: str, current_rev: Optional[str]
    ) -> List[RevisionStep]:
        with self._catch_revision_errors(
            ancestor="Destination %(end)s is not a valid downgrade "
            "target from current head(s)",
            end=destination,
        ):
            revs = self.iterate_revisions(
                current_rev, destination, select_for_downgrade=True
            )
            return [
                migration.MigrationStep.downgrade_from_script(
                    self.revision_map, script
                )
                for script in revs
            ]

    def _stamp_revs(
        self, revision: _RevIdType, heads: _RevIdType
    ) -> List[StampStep]:
        with self._catch_revision_errors(
            multiple_heads="Multiple heads are present; please specify a "
            "single target revision"
        ):
            heads_revs = self.get_revisions(heads)

            steps = []

            if not revision:
                revision = "base"

            filtered_heads: List[Script] = []
            for rev in util.to_tuple(revision):
                if rev:
                    filtered_heads.extend(
                        self.revision_map.filter_for_lineage(
                            cast(Sequence[Script], heads_revs),
                            rev,
                            include_dependencies=True,
                        )
                    )
            filtered_heads = util.unique_list(filtered_heads)

            dests = self.get_revisions(revision) or [None]

            for dest in dests:
                if dest is None:
                    # dest is 'base'.  Return a "delete branch" migration
                    # for all applicable heads.
                    steps.extend(
                        [
                            migration.StampStep(
                                head.revision,
                                None,
                                False,
                                True,
                                self.revision_map,
                            )
                            for head in filtered_heads
                        ]
                    )
                    continue
                elif dest in filtered_heads:
                    # the dest is already in the version table, do nothing.
                    continue

                # figure out if the dest is a descendant or an
                # ancestor of the selected nodes
                descendants = set(
                    self.revision_map._get_descendant_nodes([dest])
                )
                ancestors = set(self.revision_map._get_ancestor_nodes([dest]))

                if descendants.intersection(filtered_heads):
                    # heads are above the target, so this is a downgrade.
                    # we can treat them as a "merge", single step.
                    assert not ancestors.intersection(filtered_heads)
                    todo_heads = [head.revision for head in filtered_heads]
                    step = migration.StampStep(
                        todo_heads,
                        dest.revision,
                        False,
                        False,
                        self.revision_map,
                    )
                    steps.append(step)
                    continue
                elif ancestors.intersection(filtered_heads):
                    # heads are below the target, so this is an upgrade.
                    # we can treat them as a "merge", single step.
                    todo_heads = [head.revision for head in filtered_heads]
                    step = migration.StampStep(
                        todo_heads,
                        dest.revision,
                        True,
                        False,
                        self.revision_map,
                    )
                    steps.append(step)
                    continue
                else:
                    # destination is in a branch not represented,
                    # treat it as new branch
                    step = migration.StampStep(
                        (), dest.revision, True, True, self.revision_map
                    )
                    steps.append(step)
                    continue

            return steps

    def run_env(self) -> None:
        """Run the script environment.

        This basically runs the ``env.py`` script present
        in the migration environment.   It is called exclusively
        by the command functions in :mod:`alembic.command`.


        """
        util.load_python_file(self.dir, "env.py")

    @property
    def env_py_location(self) -> str:
        return str(Path(self.dir, "env.py"))

    def _append_template(self, src: Path, dest: Path, **kw: Any) -> None:
        with util.status(
            f"Appending to existing {dest.absolute()}",
            **self.messaging_opts,
        ):
            util.template_to_file(
                src,
                dest,
                self.output_encoding,
                append_with_newlines=True,
                **kw,
            )

    def _generate_template(self, src: Path, dest: Path, **kw: Any) -> None:
        with util.status(
            f"Generating {dest.absolute()}", **self.messaging_opts
        ):
            util.template_to_file(src, dest, self.output_encoding, **kw)

    def _copy_file(self, src: Path, dest: Path) -> None:
        with util.status(
            f"Generating {dest.absolute()}", **self.messaging_opts
        ):
            shutil.copy(src, dest)

    def _ensure_directory(self, path: Path) -> None:
        path = path.absolute()
        if not path.exists():
            with util.status(
                f"Creating directory {path}", **self.messaging_opts
            ):
                os.makedirs(path)

    def _generate_create_date(self) -> datetime.datetime:
        if self.timezone is not None:
            if ZoneInfo is None:
                raise util.CommandError(
                    "Python >= 3.9 is required for timezone support or "
                    "the 'backports.zoneinfo' package must be installed."
                )
            # First, assume correct capitalization
            try:
                tzinfo = ZoneInfo(self.timezone)
            except ZoneInfoNotFoundError:
                tzinfo = None
            if tzinfo is None:
                try:
                    tzinfo = ZoneInfo(self.timezone.upper())
                except ZoneInfoNotFoundError:
                    raise util.CommandError(
                        "Can't locate timezone: %s" % self.timezone
                    ) from None

            create_date = datetime.datetime.now(
                tz=datetime.timezone.utc
            ).astimezone(tzinfo)
        else:
            create_date = datetime.datetime.now()
        return create_date

    def generate_revision(
        self,
        revid: str,
        message: Optional[str],
        head: Optional[_RevIdType] = None,
        splice: Optional[bool] = False,
        branch_labels: Optional[_RevIdType] = None,
        version_path: Union[str, os.PathLike[str], None] = None,
        file_template: Optional[str] = None,
        depends_on: Optional[_RevIdType] = None,
        **kw: Any,
    ) -> Optional[Script]:
        """Generate a new revision file.

        This runs the ``script.py.mako`` template, given
        template arguments, and creates a new file.

        :param revid: String revision id.  Typically this
         comes from ``alembic.util.rev_id()``.
        :param message: the revision message, the one passed
         by the -m argument to the ``revision`` command.
        :param head: the head revision to generate against.  Defaults
         to the current "head" if no branches are present, else raises
         an exception.
        :param splice: if True, allow the "head" version to not be an
         actual head; otherwise, the selected head must be a head
         (e.g. endpoint) revision.

        """
        if head is None:
            head = "head"

        try:
            Script.verify_rev_id(revid)
        except revision.RevisionError as err:
            raise util.CommandError(err.args[0]) from err

        with self._catch_revision_errors(
            multiple_heads=(
                "Multiple heads are present; please specify the head "
                "revision on which the new revision should be based, "
                "or perform a merge."
            )
        ):
            heads = cast(
                Tuple[Optional["Revision"], ...],
                self.revision_map.get_revisions(head),
            )
            for h in heads:
                assert h != "base"  # type: ignore[comparison-overlap]

        if len(set(heads)) != len(heads):
            raise util.CommandError("Duplicate head revisions specified")

        create_date = self._generate_create_date()

        if version_path is None:
            if len(self._version_locations) > 1:
                for head_ in heads:
                    if head_ is not None:
                        assert isinstance(head_, Script)
                        version_path = head_._script_path.parent
                        break
                else:
                    raise util.CommandError(
                        "Multiple version locations present, "
                        "please specify --version-path"
                    )
            else:
                version_path = self._singular_version_location
        else:
            version_path = Path(version_path)

        assert isinstance(version_path, Path)
        norm_path = version_path.absolute()
        for vers_path in self._version_locations:
            if vers_path.absolute() == norm_path:
                break
        else:
            raise util.CommandError(
                f"Path {version_path} is not represented in current "
                "version locations"
            )

        if self.version_locations:
            self._ensure_directory(version_path)

        path = self._rev_path(version_path, revid, message, create_date)

        if not splice:
            for head_ in heads:
                if head_ is not None and not head_.is_head:
                    raise util.CommandError(
                        "Revision %s is not a head revision; please specify "
                        "--splice to create a new branch from this revision"
                        % head_.revision
                    )

        resolved_depends_on: Optional[List[str]]
        if depends_on:
            with self._catch_revision_errors():
                resolved_depends_on = [
                    (
                        dep
                        if dep in rev.branch_labels  # maintain branch labels
                        else rev.revision
                    )  # resolve partial revision identifiers
                    for rev, dep in [
                        (not_none(self.revision_map.get_revision(dep)), dep)
                        for dep in util.to_list(depends_on)
                    ]
                ]
        else:
            resolved_depends_on = None

        self._generate_template(
            Path(self.dir, "script.py.mako"),
            path,
            up_revision=str(revid),
            down_revision=revision.tuple_rev_as_scalar(
                tuple(h.revision if h is not None else None for h in heads)
            ),
            branch_labels=util.to_tuple(branch_labels),
            depends_on=revision.tuple_rev_as_scalar(resolved_depends_on),
            create_date=create_date,
            comma=util.format_as_comma,
            message=message if message is not None else ("empty message"),
            **kw,
        )

        post_write_hooks = self.hooks
        if post_write_hooks:
            write_hooks._run_hooks(path, post_write_hooks)

        try:
            script = Script._from_path(self, path)
        except revision.RevisionError as err:
            raise util.CommandError(err.args[0]) from err
        if script is None:
            return None
        if branch_labels and not script.branch_labels:
            raise util.CommandError(
                "Version %s specified branch_labels %s, however the "
                "migration file %s does not have them; have you upgraded "
                "your script.py.mako to include the "
                "'branch_labels' section?"
                % (script.revision, branch_labels, script.path)
            )
        self.revision_map.add_revision(script)
        return script

    def _rev_path(
        self,
        path: Union[str, os.PathLike[str]],
        rev_id: str,
        message: Optional[str],
        create_date: datetime.datetime,
    ) -> Path:
        epoch = int(create_date.timestamp())
        slug = "_".join(_slug_re.findall(message or "")).lower()
        if len(slug) > self.truncate_slug_length:
            slug = slug[: self.truncate_slug_length].rsplit("_", 1)[0] + "_"
        filename = "%s.py" % (
            self.file_template
            % {
                "rev": rev_id,
                "slug": slug,
                "epoch": epoch,
                "year": create_date.year,
                "month": create_date.month,
                "day": create_date.day,
                "hour": create_date.hour,
                "minute": create_date.minute,
                "second": create_date.second,
            }
        )
        return Path(path) / filename


class Script(revision.Revision):
    """Represent a single revision file in a ``versions/`` directory.

    The :class:`.Script` instance is returned by methods
    such as :meth:`.ScriptDirectory.iterate_revisions`.

    """

    def __init__(
        self,
        module: ModuleType,
        rev_id: str,
        path: Union[str, os.PathLike[str]],
    ):
        self.module = module
        self.path = _preserving_path_as_str(path)
        super().__init__(
            rev_id,
            module.down_revision,
            branch_labels=util.to_tuple(
                getattr(module, "branch_labels", None), default=()
            ),
            dependencies=util.to_tuple(
                getattr(module, "depends_on", None), default=()
            ),
        )

    module: ModuleType
    """The Python module representing the actual script itself."""

    path: str
    """Filesystem path of the script."""

    @property
    def _script_path(self) -> Path:
        return Path(self.path)

    _db_current_indicator: Optional[bool] = None
    """Utility variable which when set will cause string output to indicate
    this is a "current" version in some database"""

    @property
    def doc(self) -> str:
        """Return the docstring given in the script."""

        return re.split("\n\n", self.longdoc)[0]

    @property
    def longdoc(self) -> str:
        """Return the docstring given in the script."""

        doc = self.module.__doc__
        if doc:
            if hasattr(self.module, "_alembic_source_encoding"):
                doc = doc.decode(  # type: ignore[attr-defined]
                    self.module._alembic_source_encoding
                )
            return doc.strip()
        else:
            return ""

    @property
    def log_entry(self) -> str:
        entry = "Rev: %s%s%s%s%s\n" % (
            self.revision,
            " (head)" if self.is_head else "",
            " (branchpoint)" if self.is_branch_point else "",
            " (mergepoint)" if self.is_merge_point else "",
            " (current)" if self._db_current_indicator else "",
        )
        if self.is_merge_point:
            entry += "Merges: %s\n" % (self._format_down_revision(),)
        else:
            entry += "Parent: %s\n" % (self._format_down_revision(),)

        if self.dependencies:
            entry += "Also depends on: %s\n" % (
                util.format_as_comma(self.dependencies)
            )

        if self.is_branch_point:
            entry += "Branches into: %s\n" % (
                util.format_as_comma(self.nextrev)
            )

        if self.branch_labels:
            entry += "Branch names: %s\n" % (
                util.format_as_comma(self.branch_labels),
            )

        entry += "Path: %s\n" % (self.path,)

        entry += "\n%s\n" % (
            "\n".join("    %s" % para for para in self.longdoc.splitlines())
        )
        return entry

    def __str__(self) -> str:
        return "%s -> %s%s%s%s, %s" % (
            self._format_down_revision(),
            self.revision,
            " (head)" if self.is_head else "",
            " (branchpoint)" if self.is_branch_point else "",
            " (mergepoint)" if self.is_merge_point else "",
            self.doc,
        )

    def _head_only(
        self,
        include_branches: bool = False,
        include_doc: bool = False,
        include_parents: bool = False,
        tree_indicators: bool = True,
        head_indicators: bool = True,
    ) -> str:
        text = self.revision
        if include_parents:
            if self.dependencies:
                text = "%s (%s) -> %s" % (
                    self._format_down_revision(),
                    util.format_as_comma(self.dependencies),
                    text,
                )
            else:
                text = "%s -> %s" % (self._format_down_revision(), text)
        assert text is not None
        if include_branches and self.branch_labels:
            text += " (%s)" % util.format_as_comma(self.branch_labels)
        if head_indicators or tree_indicators:
            text += "%s%s%s" % (
                " (head)" if self._is_real_head else "",
                (
                    " (effective head)"
                    if self.is_head and not self._is_real_head
                    else ""
                ),
                " (current)" if self._db_current_indicator else "",
            )
        if tree_indicators:
            text += "%s%s" % (
                " (branchpoint)" if self.is_branch_point else "",
                " (mergepoint)" if self.is_merge_point else "",
            )
        if include_doc:
            text += ", %s" % self.doc
        return text

    def cmd_format(
        self,
        verbose: bool,
        include_branches: bool = False,
        include_doc: bool = False,
        include_parents: bool = False,
        tree_indicators: bool = True,
    ) -> str:
        if verbose:
            return self.log_entry
        else:
            return self._head_only(
                include_branches, include_doc, include_parents, tree_indicators
            )

    def _format_down_revision(self) -> str:
        if not self.down_revision:
            return "<base>"
        else:
            return util.format_as_comma(self._versioned_down_revisions)

    @classmethod
    def _list_py_dir(
        cls, scriptdir: ScriptDirectory, path: Path
    ) -> List[Path]:
        paths = []
        for root, dirs, files in compat.path_walk(path, top_down=True):
            if root.name.endswith("__pycache__"):
                # a special case - we may include these files
                # if a `sourceless` option is specified
                continue

            for filename in sorted(files):
                paths.append(root / filename)

            if scriptdir.sourceless:
                # look for __pycache__
                py_cache_path = root / "__pycache__"
                if py_cache_path.exists():
                    # add all files from __pycache__ whose filename is not
                    # already in the names we got from the version directory.
                    # add as relative paths including __pycache__ token
                    names = {
                        Path(filename).name.split(".")[0] for filename in files
                    }
                    paths.extend(
                        py_cache_path / pyc
                        for pyc in py_cache_path.iterdir()
                        if pyc.name.split(".")[0] not in names
                    )

            if not scriptdir.recursive_version_locations:
                break

            # the real script order is defined by revision,
            # but it may be undefined if there are many files with a same
            # `down_revision`, for a better user experience (ex. debugging),
            # we use a deterministic order
            dirs.sort()

        return paths

    @classmethod
    def _from_path(
        cls, scriptdir: ScriptDirectory, path: Union[str, os.PathLike[str]]
    ) -> Optional[Script]:

        path = Path(path)
        dir_, filename = path.parent, path.name

        if scriptdir.sourceless:
            py_match = _sourceless_rev_file.match(filename)
        else:
            py_match = _only_source_rev_file.match(filename)

        if not py_match:
            return None

        py_filename = py_match.group(1)

        if scriptdir.sourceless:
            is_c = py_match.group(2) == "c"
            is_o = py_match.group(2) == "o"
        else:
            is_c = is_o = False

        if is_o or is_c:
            py_exists = (dir_ / py_filename).exists()
            pyc_exists = (dir_ / (py_filename + "c")).exists()

            # prefer .py over .pyc because we'd like to get the
            # source encoding; prefer .pyc over .pyo because we'd like to
            # have the docstrings which a -OO file would not have
            if py_exists or is_o and pyc_exists:
                return None

        module = util.load_python_file(dir_, filename)

        if not hasattr(module, "revision"):
            # attempt to get the revision id from the script name,
            # this for legacy only
            m = _legacy_rev.match(filename)
            if not m:
                raise util.CommandError(
                    "Could not determine revision id from "
                    f"filename {filename}. "
                    "Be sure the 'revision' variable is "
                    "declared inside the script (please see 'Upgrading "
                    "from Alembic 0.1 to 0.2' in the documentation)."
                )
            else:
                revision = m.group(1)
        else:
            revision = module.revision
        return Script(module, revision, dir_ / filename)
