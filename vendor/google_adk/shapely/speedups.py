import warnings

__all__ = ["available", "enable", "disable", "enabled"]


available = True
enabled = True


_MSG = (
    "This function has no longer any effect, and will be removed in a "
    "future release. Starting with Shapely 2.0, equivalent speedups are "
    "always available"
)


def enable():
    """
    This function has no longer any effect, and will be removed in a future
    release.

    Previously, this function enabled cython-based speedups. Starting with
    Shapely 2.0, equivalent speedups are available in every installation.
    """
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)


def disable():
    """
    This function has no longer any effect, and will be removed in a future
    release.

    Previously, this function enabled cython-based speedups. Starting with
    Shapely 2.0, equivalent speedups are available in every installation.
    """
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)
