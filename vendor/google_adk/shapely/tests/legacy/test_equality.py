import pytest

from shapely import Point
from shapely.errors import ShapelyDeprecationWarning


def test_equals_exact():
    p1 = Point(1.0, 1.0)
    p2 = Point(2.0, 2.0)
    assert not p1.equals(p2)
    assert not p1.equals_exact(p2, 0.001)
    assert p1.equals_exact(p2, 1.42)


def test_almost_equals_default():
    p1 = Point(1.0, 1.0)
    p2 = Point(1.0 + 1e-7, 1.0 + 1e-7)  # almost equal to 6 places
    p3 = Point(1.0 + 1e-6, 1.0 + 1e-6)  # not almost equal
    with pytest.warns(ShapelyDeprecationWarning):
        assert p1.almost_equals(p2)
    with pytest.warns(ShapelyDeprecationWarning):
        assert not p1.almost_equals(p3)


def test_almost_equals():
    p1 = Point(1.0, 1.0)
    p2 = Point(1.1, 1.1)
    assert not p1.equals(p2)
    with pytest.warns(ShapelyDeprecationWarning):
        assert p1.almost_equals(p2, 0)
    with pytest.warns(ShapelyDeprecationWarning):
        assert not p1.almost_equals(p2, 1)
