[tool.alembic]

# path to migration scripts.
# this is typically a path given in POSIX (e.g. forward slashes)
# format, relative to the token %(here)s which refers to the location of this
# ini file
script_location = "${script_location}"

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
# Uncomment the line below if you want the files to be prepended with date and time
# see https://alembic.sqlalchemy.org/en/latest/tutorial.html#editing-the-ini-file
# for all available tokens
# file_template = "%%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s"

# additional paths to be prepended to sys.path. defaults to the current working directory.
prepend_sys_path = [
    "."
]

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python>=3.9 or backports.zoneinfo library and tzdata library.
# Any required deps can installed by adding `alembic[tz]` to the pip requirements
# string value is passed to ZoneInfo()
# leave blank for localtime
# timezone =

# max length of characters to apply to the "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; This defaults
# to <script_location>/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path.
# version_locations = [
#    "%(here)s/alembic/versions",
#    "%(here)s/foo/bar"
# ]


# set to 'true' to search source files recursively
# in each "version_locations" directory
# new in Alembic version 1.10
# recursive_version_locations = false

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = "utf-8"

# This section defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples
# [[tool.alembic.post_write_hooks]]
# format using "black" - use the console_scripts runner,
# against the "black" entrypoint
# name = "black"
# type = "console_scripts"
# entrypoint = "black"
# options = "-l 79 REVISION_SCRIPT_FILENAME"
#
# [[tool.alembic.post_write_hooks]]
# lint with attempts to fix using "ruff" - use the module runner, against the "ruff" module
# name = "ruff"
# type = "module"
# module = "ruff"
# options = "check --fix REVISION_SCRIPT_FILENAME"
#
# [[tool.alembic.post_write_hooks]]
# Alternatively, use the exec runner to execute a binary found on your PATH
# name = "ruff"
# type = "exec"
# executable = "ruff"
# options = "check --fix REVISION_SCRIPT_FILENAME"

