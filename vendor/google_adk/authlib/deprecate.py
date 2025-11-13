import warnings


class AuthlibDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("always", AuthlibDeprecationWarning)


def deprecate(message, version=None, link_uid=None, link_file=None, stacklevel=3):
    if version:
        message += f"\nIt will be compatible before version {version}."

    if link_uid and link_file:
        message += f"\nRead more <https://git.io/{link_uid}#file-{link_file}-md>"

    warnings.warn(AuthlibDeprecationWarning(message), stacklevel=stacklevel)
