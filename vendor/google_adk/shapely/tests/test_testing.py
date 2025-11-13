import numpy as np
import pytest

import shapely
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    all_types,
    empty,
    empty_line_string,
    empty_line_string_z,
    empty_point,
    empty_point_z,
    empty_polygon,
    line_string,
    line_string_nan,
    line_string_z,
    point,
)

EMPTY_GEOMS = (
    empty_point,
    empty_point_z,
    empty_line_string,
    empty_line_string_z,
    empty_polygon,
    empty,
)

line_string_reversed = shapely.linestrings([(0, 0), (1, 0), (1, 1)][::-1])


PRE_GEOS_390 = pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="2D and 3D empty geometries did not have dimensionality before GEOS 3.9",
)


def make_array(left, right, use_array):
    if use_array in ("left", "both"):
        left = np.array([left] * 3, dtype=object)
    if use_array in ("right", "both"):
        right = np.array([right] * 3, dtype=object)
    return left, right


@pytest.mark.parametrize("use_array", ["none", "left", "right", "both"])
@pytest.mark.parametrize("geom", all_types + EMPTY_GEOMS)
def test_assert_geometries_equal(geom, use_array):
    assert_geometries_equal(*make_array(geom, geom, use_array))


@pytest.mark.parametrize("use_array", ["none", "left", "right", "both"])
@pytest.mark.parametrize(
    "geom1,geom2",
    [
        (point, line_string),
        (line_string, line_string_z),
        (empty_point, empty_polygon),
        pytest.param(empty_point, empty_point_z, marks=PRE_GEOS_390),
        pytest.param(empty_line_string, empty_line_string_z, marks=PRE_GEOS_390),
    ],
)
def test_assert_geometries_not_equal(geom1, geom2, use_array):
    with pytest.raises(AssertionError):
        assert_geometries_equal(*make_array(geom1, geom2, use_array))


@pytest.mark.parametrize("use_array", ["none", "left", "right", "both"])
def test_assert_none_equal(use_array):
    assert_geometries_equal(*make_array(None, None, use_array))


@pytest.mark.parametrize("use_array", ["none", "left", "right", "both"])
def test_assert_none_not_equal(use_array):
    with pytest.raises(AssertionError):
        assert_geometries_equal(*make_array(None, None, use_array), equal_none=False)


@pytest.mark.parametrize("use_array", ["none", "left", "right", "both"])
def test_assert_nan_equal(use_array):
    assert_geometries_equal(*make_array(line_string_nan, line_string_nan, use_array))


@pytest.mark.parametrize("use_array", ["none", "left", "right", "both"])
def test_assert_nan_not_equal(use_array):
    with pytest.raises(AssertionError):
        assert_geometries_equal(
            *make_array(line_string_nan, line_string_nan, use_array), equal_nan=False
        )


def test_normalize_true():
    assert_geometries_equal(line_string_reversed, line_string, normalize=True)


def test_normalize_default():
    with pytest.raises(AssertionError):
        assert_geometries_equal(line_string_reversed, line_string)


def test_normalize_false():
    with pytest.raises(AssertionError):
        assert_geometries_equal(line_string_reversed, line_string, normalize=False)
