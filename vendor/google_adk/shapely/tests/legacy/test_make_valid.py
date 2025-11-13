from shapely.geometry import Polygon
from shapely.tests.legacy.conftest import requires_geos_38
from shapely.validation import make_valid


@requires_geos_38
def test_make_valid_invalid_input():
    geom = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
    valid = make_valid(geom)
    assert len(valid.geoms) == 2
    assert all(geom.geom_type == "Polygon" for geom in valid.geoms)


@requires_geos_38
def test_make_valid_input():
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    valid = make_valid(geom)
    assert id(valid) == id(geom)
