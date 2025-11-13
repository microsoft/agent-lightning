import numpy
import pytest

from shapely.geos import geos_version

requires_geos_38 = pytest.mark.skipif(
    geos_version < (3, 8, 0), reason="GEOS >= 3.8.0 is required."
)
requires_geos_342 = pytest.mark.skipif(
    geos_version < (3, 4, 2), reason="GEOS > 3.4.2 is required."
)

shapely20_todo = pytest.mark.xfail(
    strict=True, reason="Not yet implemented for Shapely 2.0"
)
shapely20_wontfix = pytest.mark.xfail(strict=True, reason="Will fail for Shapely 2.0")


def pytest_report_header(config):
    """Header for pytest."""
    return f"dependencies: numpy-{numpy.__version__}"
