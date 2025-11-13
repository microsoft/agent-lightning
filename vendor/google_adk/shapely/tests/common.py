from contextlib import contextmanager

import numpy as np
import pytest

import shapely

shapely20_todo = pytest.mark.xfail(
    strict=False, reason="Not yet implemented for Shapely 2.0"
)

point_polygon_testdata = (
    shapely.points(np.arange(6), np.arange(6)),
    shapely.box(2, 2, 4, 4),
)
point = shapely.Point(2, 3)
line_string = shapely.LineString([(0, 0), (1, 0), (1, 1)])
linear_ring = shapely.LinearRing([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
polygon = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
multi_point = shapely.MultiPoint([(0, 0), (1, 2)])
multi_line_string = shapely.MultiLineString([[(0, 0), (1, 2)]])
multi_polygon = shapely.multipolygons(
    [
        [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        [(2.1, 2.1), (2.2, 2.1), (2.2, 2.2), (2.1, 2.2), (2.1, 2.1)],
    ]
)
geometry_collection = shapely.GeometryCollection(
    [shapely.Point(51, -1), shapely.LineString([(52, -1), (49, 2)])]
)
point_z = shapely.Point(2, 3, 4)
line_string_z = shapely.LineString([(0, 0, 4), (1, 0, 4), (1, 1, 4)])
polygon_z = shapely.Polygon([(0, 0, 4), (2, 0, 4), (2, 2, 4), (0, 2, 4), (0, 0, 4)])
geometry_collection_z = shapely.GeometryCollection([point_z, line_string_z])
polygon_with_hole = shapely.Polygon(
    [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
)
empty_point = shapely.from_wkt("POINT EMPTY")
empty_point_z = shapely.from_wkt("POINT Z EMPTY")
empty_line_string = shapely.from_wkt("LINESTRING EMPTY")
empty_line_string_z = shapely.from_wkt("LINESTRING Z EMPTY")
empty_polygon = shapely.from_wkt("POLYGON EMPTY")
empty = shapely.from_wkt("GEOMETRYCOLLECTION EMPTY")
multi_point_z = shapely.MultiPoint([(0, 0, 4), (1, 2, 4)])
multi_line_string_z = shapely.MultiLineString([[(0, 0, 4), (1, 2, 4)]])
multi_polygon_z = shapely.multipolygons(
    [
        [(0, 0, 4), (1, 0, 4), (1, 1, 4), (0, 1, 4), (0, 0, 4)],
        [(2.1, 2.1, 4), (2.2, 2.1, 4), (2.2, 2.2, 4), (2.1, 2.2, 4), (2.1, 2.1, 4)],
    ]
)
polygon_with_hole_z = shapely.Polygon(
    [(0, 0, 4), (0, 10, 4), (10, 10, 4), (10, 0, 4), (0, 0, 4)],
    holes=[[(2, 2, 4), (2, 4, 4), (4, 4, 4), (4, 2, 4), (2, 2, 4)]],
)

all_types = (
    point,
    line_string,
    linear_ring,
    polygon,
    multi_point,
    multi_line_string,
    multi_polygon,
    geometry_collection,
    empty,
)

all_types_z = (
    point_z,
    line_string_z,
    polygon_z,
    multi_point_z,
    multi_line_string_z,
    multi_polygon_z,
    polygon_with_hole_z,
    geometry_collection_z,
    empty_point_z,
    empty_line_string_z,
)


@contextmanager
def ignore_invalid(condition=True):
    if condition:
        with np.errstate(invalid="ignore"):
            yield
    else:
        yield


with ignore_invalid():
    line_string_nan = shapely.LineString([(np.nan, np.nan), (np.nan, np.nan)])


class ArrayLike:
    """
    Simple numpy Array like class that implements the
    ufunc protocol.
    """

    def __init__(self, array):
        self._array = np.asarray(array)

    def __len__(self):
        return len(self._array)

    def __getitem(self, key):
        return self._array[key]

    def __iter__(self):
        return self._array.__iter__()

    def __array__(self):
        return np.asarray(self._array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            inputs = [
                arg._array if isinstance(arg, self.__class__) else arg for arg in inputs
            ]
            return self.__class__(ufunc(*inputs, **kwargs))
        else:
            return NotImplemented
