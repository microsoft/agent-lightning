import platform
import weakref

import numpy as np
import pytest

import shapely
from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal


def test_polygon():
    assert bool(Polygon()) is False


def test_linestring():
    assert bool(LineString()) is False


def test_point():
    assert bool(Point()) is False


def test_geometry_collection():
    assert bool(GeometryCollection()) is False


geometries_all_types = [
    Point(1, 1),
    LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]),
    LineString([(0, 0), (1, 1), (0, 1), (0, 0)]),
    Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    MultiPoint([(1, 1)]),
    MultiLineString([[(0, 0), (1, 1), (0, 1), (0, 0)]]),
    MultiPolygon([Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])]),
    GeometryCollection([Point(1, 1)]),
]


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="Setting custom attributes doesn't fail on PyPy",
)
@pytest.mark.parametrize("geom", geometries_all_types)
def test_setattr_disallowed(geom):
    with pytest.raises(AttributeError):
        geom.name = "test"


@pytest.mark.parametrize("geom", geometries_all_types)
def test_weakrefable(geom):
    _ = weakref.ref(geom)


def test_base_class_not_callable():
    with pytest.raises(TypeError):
        shapely.Geometry("POINT (1 1)")


def test_GeometryType_deprecated():
    geom = Point(1, 1)

    with pytest.warns(ShapelyDeprecationWarning):
        geom_type = geom.geometryType()

    assert geom_type == geom.geom_type


def test_type_deprecated():
    geom = Point(1, 1)

    with pytest.warns(ShapelyDeprecationWarning):
        geom_type = geom.type

    assert geom_type == geom.geom_type


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_segmentize():
    line = LineString([(0, 0), (0, 10)])
    result = line.segmentize(max_segment_length=5)
    assert result.equals(LineString([(0, 0), (0, 5), (0, 10)]))


@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_reverse():
    coords = [(0, 0), (1, 2)]
    line = LineString(coords)
    result = line.reverse()
    assert result.coords[:] == coords[::-1]


@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize(
    "op", ["union", "intersection", "difference", "symmetric_difference"]
)
@pytest.mark.parametrize("grid_size", [0, 1, 2])
def test_binary_op_grid_size(op, grid_size):
    geom1 = shapely.box(0, 0, 2.5, 2.5)
    geom2 = shapely.box(2, 2, 3, 3)

    result = getattr(geom1, op)(geom2, grid_size=grid_size)
    expected = getattr(shapely, op)(geom1, geom2, grid_size=grid_size)
    assert result == expected


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_dwithin():
    point = Point(1, 1)
    line = LineString([(0, 0), (0, 10)])
    assert point.dwithin(line, 0.5) is False
    assert point.dwithin(line, 1.5) is True


def test_contains_properly():
    polygon = Polygon([(0, 0), (10, 10), (10, -10)])
    line = LineString([(0, 0), (10, 0)])
    assert polygon.contains_properly(line) is False
    assert polygon.contains(line) is True


@pytest.mark.parametrize(
    "op", ["convex_hull", "envelope", "oriented_envelope", "minimum_rotated_rectangle"]
)
def test_constructive_properties(op):
    geom = LineString([(0, 0), (0, 10), (10, 10)])
    result = getattr(geom, op)
    expected = getattr(shapely, op)(geom)
    assert result == expected


@pytest.mark.parametrize(
    "op",
    [
        "crosses",
        "contains",
        "contains_properly",
        "covered_by",
        "covers",
        "disjoint",
        "equals",
        "intersects",
        "overlaps",
        "touches",
        "within",
    ],
)
def test_array_argument_binary_predicates(op):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])

    result = getattr(polygon, op)(points)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(polygon, op)(p) for p in points], dtype=bool)
    np.testing.assert_array_equal(result, expected)

    # check scalar
    result = getattr(polygon, op)(points[0])
    assert type(result) is bool


@pytest.mark.parametrize(
    "op, kwargs",
    [
        pytest.param(
            "dwithin",
            dict(distance=0.5),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10"
            ),
        ),
        ("equals_exact", dict(tolerance=0.01)),
        ("relate_pattern", dict(pattern="T*F**F***")),
    ],
)
def test_array_argument_binary_predicates2(op, kwargs):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])

    result = getattr(polygon, op)(points, **kwargs)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(polygon, op)(p, **kwargs) for p in points], dtype=bool)
    np.testing.assert_array_equal(result, expected)

    # check scalar
    result = getattr(polygon, op)(points[0], **kwargs)
    assert type(result) is bool


@pytest.mark.parametrize(
    "op",
    [
        "difference",
        "intersection",
        "symmetric_difference",
        "union",
    ],
)
def test_array_argument_binary_geo(op):
    box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    polygons = shapely.buffer(shapely.points([(0, 0), (0.5, 0.5), (1, 1)]), 0.5)

    result = getattr(box, op)(polygons)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(box, op)(g) for g in polygons], dtype=object)
    assert_geometries_equal(result, expected)

    # check scalar
    result = getattr(box, op)(polygons[0])
    assert isinstance(result, (Polygon, MultiPolygon))


@pytest.mark.parametrize("op", ["distance", "hausdorff_distance"])
def test_array_argument_float(op):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])

    result = getattr(polygon, op)(points)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(polygon, op)(p) for p in points], dtype="float64")
    np.testing.assert_array_equal(result, expected)

    # check scalar
    result = getattr(polygon, op)(points[0])
    assert type(result) is float


@pytest.mark.parametrize("op", ["line_interpolate_point", "interpolate"])
def test_array_argument_linear_point(op):
    line = LineString([(0, 0), (0, 1), (1, 1)])
    distances = np.array([0, 0.5, 1])

    result = getattr(line, op)(distances)
    assert isinstance(result, np.ndarray)
    expected = np.array(
        [line.line_interpolate_point(d) for d in distances], dtype=object
    )
    assert_geometries_equal(result, expected)

    # check scalar
    result = getattr(line, op)(distances[0])
    assert isinstance(result, Point)


@pytest.mark.parametrize("op", ["line_locate_point", "project"])
def test_array_argument_linear_float(op):
    line = LineString([(0, 0), (0, 1), (1, 1)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])

    result = getattr(line, op)(points)
    assert isinstance(result, np.ndarray)
    expected = np.array([line.line_locate_point(p) for p in points], dtype="float64")
    np.testing.assert_array_equal(result, expected)

    # check scalar
    result = getattr(line, op)(points[0])
    assert type(result) is float


def test_array_argument_buffer():
    point = Point(1, 1)
    distances = np.array([0, 0.5, 1])

    result = point.buffer(distances)
    assert isinstance(result, np.ndarray)
    expected = np.array([point.buffer(d) for d in distances], dtype=object)
    assert_geometries_equal(result, expected)

    # check scalar
    result = point.buffer(distances[0])
    assert isinstance(result, Polygon)
