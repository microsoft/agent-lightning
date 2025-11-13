import warnings

import numpy as np

from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos

__all__ = [
    "has_z",
    "is_ccw",
    "is_closed",
    "is_empty",
    "is_geometry",
    "is_missing",
    "is_prepared",
    "is_ring",
    "is_simple",
    "is_valid",
    "is_valid_input",
    "is_valid_reason",
    "crosses",
    "contains",
    "contains_xy",
    "contains_properly",
    "covered_by",
    "covers",
    "disjoint",
    "dwithin",
    "equals",
    "intersects",
    "intersects_xy",
    "overlaps",
    "touches",
    "within",
    "equals_exact",
    "relate",
    "relate_pattern",
]


@multithreading_enabled
def has_z(geometry, **kwargs):
    """Returns True if a geometry has a Z coordinate.

    Note that this function returns False if the (first) Z coordinate equals NaN or
    if the geometry is empty.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_coordinate_dimension

    Examples
    --------
    >>> from shapely import Point
    >>> has_z(Point(0, 0))
    False
    >>> has_z(Point(0, 0, 0))
    True
    >>> has_z(Point(0, 0, float("nan")))
    False
    """
    return lib.has_z(geometry, **kwargs)


@requires_geos("3.7.0")
@multithreading_enabled
def is_ccw(geometry, **kwargs):
    """Returns True if a linestring or linearring is counterclockwise.

    Note that there are no checks on whether lines are actually closed and
    not self-intersecting, while this is a requirement for is_ccw. The recommended
    usage of this function for linestrings is ``is_ccw(g) & is_simple(g)`` and for
    linearrings ``is_ccw(g) & is_valid(g)``.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linear geometries and for
        lines with fewer than 4 points (including the closing point).
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_simple : Checks if a linestring is closed and simple.
    is_valid : Checks additionally if the geometry is simple.

    Examples
    --------
    >>> from shapely import LinearRing, LineString, Point
    >>> is_ccw(LinearRing([(0, 0), (0, 1), (1, 1), (0, 0)]))
    False
    >>> is_ccw(LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]))
    True
    >>> is_ccw(LineString([(0, 0), (1, 1), (0, 1)]))
    False
    >>> is_ccw(Point(0, 0))
    False
    """
    return lib.is_ccw(geometry, **kwargs)


@multithreading_enabled
def is_closed(geometry, **kwargs):
    """Returns True if a linestring's first and last points are equal.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_ring : Checks additionally if the geometry is simple.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> is_closed(LineString([(0, 0), (1, 1)]))
    False
    >>> is_closed(LineString([(0, 0), (0, 1), (1, 1), (0, 0)]))
    True
    >>> is_closed(Point(0, 0))
    False
    """
    return lib.is_closed(geometry, **kwargs)


@multithreading_enabled
def is_empty(geometry, **kwargs):
    """Returns True if a geometry is an empty point, polygon, etc.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_missing : checks if the object is a geometry

    Examples
    --------
    >>> from shapely import Point
    >>> is_empty(Point())
    True
    >>> is_empty(Point(0, 0))
    False
    >>> is_empty(None)
    False
    """
    return lib.is_empty(geometry, **kwargs)


@multithreading_enabled
def is_geometry(geometry, **kwargs):
    """Returns True if the object is a geometry

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_missing : check if an object is missing (None)
    is_valid_input : check if an object is a geometry or None

    Examples
    --------
    >>> from shapely import GeometryCollection, Point
    >>> is_geometry(Point(0, 0))
    True
    >>> is_geometry(GeometryCollection())
    True
    >>> is_geometry(None)
    False
    >>> is_geometry("text")
    False
    """
    return lib.is_geometry(geometry, **kwargs)


@multithreading_enabled
def is_missing(geometry, **kwargs):
    """Returns True if the object is not a geometry (None)

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_geometry : check if an object is a geometry
    is_valid_input : check if an object is a geometry or None
    is_empty : checks if the object is an empty geometry

    Examples
    --------
    >>> from shapely import GeometryCollection, Point
    >>> is_missing(Point(0, 0))
    False
    >>> is_missing(GeometryCollection())
    False
    >>> is_missing(None)
    True
    >>> is_missing("text")
    False
    """
    return lib.is_missing(geometry, **kwargs)


@multithreading_enabled
def is_prepared(geometry, **kwargs):
    """Returns True if a Geometry is prepared.

    Note that it is not necessary to check if a geometry is already prepared
    before preparing it. It is more efficient to call ``prepare`` directly
    because it will skip geometries that are already prepared.

    This function will return False for missing geometries (None).

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_valid_input : check if an object is a geometry or None
    prepare : prepare a geometry

    Examples
    --------
    >>> from shapely import Point, prepare
    >>> geometry = Point(0, 0)
    >>> is_prepared(Point(0, 0))
    False
    >>> prepare(geometry)
    >>> is_prepared(geometry)
    True
    >>> is_prepared(None)
    False
    """
    return lib.is_prepared(geometry, **kwargs)


@multithreading_enabled
def is_valid_input(geometry, **kwargs):
    """Returns True if the object is a geometry or None

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_geometry : checks if an object is a geometry
    is_missing : checks if an object is None

    Examples
    --------
    >>> from shapely import GeometryCollection, Point
    >>> is_valid_input(Point(0, 0))
    True
    >>> is_valid_input(GeometryCollection())
    True
    >>> is_valid_input(None)
    True
    >>> is_valid_input(1.0)
    False
    >>> is_valid_input("text")
    False
    """
    return lib.is_valid_input(geometry, **kwargs)


@multithreading_enabled
def is_ring(geometry, **kwargs):
    """Returns True if a linestring is closed and simple.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_closed : Checks only if the geometry is closed.
    is_simple : Checks only if the geometry is simple.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> is_ring(Point(0, 0))
    False
    >>> geom = LineString([(0, 0), (1, 1)])
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (False, True, False)
    >>> geom = LineString([(0, 0), (0, 1), (1, 1), (0, 0)])
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, True, True)
    >>> geom = LineString([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, False, False)
    """
    return lib.is_ring(geometry, **kwargs)


@multithreading_enabled
def is_simple(geometry, **kwargs):
    """Returns True if a Geometry has no anomalous geometric points, such as
    self-intersections or self tangency.

    Note that polygons and linearrings are assumed to be simple. Use is_valid
    to check these kind of geometries for self-intersections.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for geometrycollections.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_ring : Checks additionally if the geometry is closed.
    is_valid : Checks whether a geometry is well formed.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> is_simple(Polygon([(1, 1), (2, 1), (2, 2), (1, 1)]))
    True
    >>> is_simple(LineString([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)]))
    False
    >>> is_simple(None)
    False
    """
    return lib.is_simple(geometry, **kwargs)


@multithreading_enabled
def is_valid(geometry, **kwargs):
    """Returns True if a geometry is well formed.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns False for missing values.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_valid_reason : Returns the reason in case of invalid.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, Polygon
    >>> is_valid(LineString([(0, 0), (1, 1)]))
    True
    >>> is_valid(Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]))
    False
    >>> is_valid(GeometryCollection())
    True
    >>> is_valid(None)
    False
    """
    # GEOS is valid will emit warnings for invalid geometries. Suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = lib.is_valid(geometry, **kwargs)
    return result


def is_valid_reason(geometry, **kwargs):
    """Returns a string stating if a geometry is valid and if not, why.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns None for missing values.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_valid : returns True or False

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> is_valid_reason(LineString([(0, 0), (1, 1)]))
    'Valid Geometry'
    >>> is_valid_reason(Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]))
    'Ring Self-intersection[1 1]'
    >>> is_valid_reason(None) is None
    True
    """
    return lib.is_valid_reason(geometry, **kwargs)


@multithreading_enabled
def crosses(a, b, **kwargs):
    """Returns True if A and B spatially cross.

    A crosses B if they have some but not all interior points in common,
    the intersection is one dimension less than the maximum dimension of A or B,
    and the intersection is not equal to either A or B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, Point, Polygon
    >>> line = LineString([(0, 0), (1, 1)])
    >>> # A contains B:
    >>> crosses(line, Point(0.5, 0.5))
    False
    >>> # A and B intersect at a point but do not share all points:
    >>> crosses(line, MultiPoint([(0, 1), (0.5, 0.5)]))
    True
    >>> crosses(line, LineString([(0, 1), (1, 0)]))
    True
    >>> # A is contained by B; their intersection is a line (same dimension):
    >>> crosses(line, LineString([(0, 0), (2, 2)]))
    False
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> # A contains B:
    >>> crosses(area, line)
    False
    >>> # A and B intersect with a line (lower dimension) but do not share all points:
    >>> crosses(area, LineString([(0, 0), (2, 2)]))
    True
    >>> # A contains B:
    >>> crosses(area, Point(0.5, 0.5))
    False
    >>> # A contains some but not all points of B; they intersect at a point:
    >>> crosses(area, MultiPoint([(2, 2), (0.5, 0.5)]))
    True
    """
    return lib.crosses(a, b, **kwargs)


@multithreading_enabled
def contains(a, b, **kwargs):
    """Returns True if geometry B is completely inside geometry A.

    A contains B if no points of B lie in the exterior of A and at least one
    point of the interior of B lies in the interior of A.

    Note: following this definition, a geometry does not contain its boundary,
    but it does contain itself. See ``contains_properly`` for a version where
    a geometry does not contain itself.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    within : ``contains(A, B) == within(B, A)``
    contains_properly : contains with no common boundary points
    prepare : improve performance by preparing ``a`` (the first argument)
    contains_xy : variant for checking against a Point with x, y coordinates

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> line = LineString([(0, 0), (1, 1)])
    >>> contains(line, Point(0, 0))
    False
    >>> contains(line, Point(0.5, 0.5))
    True
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> contains(area, Point(0, 0))
    False
    >>> contains(area, line)
    True
    >>> contains(area, LineString([(0, 0), (2, 2)]))
    False
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> contains(polygon_with_hole, Point(1, 1))
    True
    >>> contains(polygon_with_hole, Point(2, 2))
    False
    >>> contains(polygon_with_hole, LineString([(1, 1), (5, 5)]))
    False
    >>> contains(area, area)
    True
    >>> contains(area, None)
    False
    """
    return lib.contains(a, b, **kwargs)


@multithreading_enabled
def contains_properly(a, b, **kwargs):
    """Returns True if geometry B is completely inside geometry A, with no
    common boundary points.

    A contains B properly if B intersects the interior of A but not the
    boundary (or exterior). This means that a geometry A does not
    "contain properly" itself, which contrasts with the ``contains`` function,
    where common points on the boundary are allowed.

    Note: this function will prepare the geometries under the hood if needed.
    You can prepare the geometries in advance to avoid repeated preparation
    when calling this function multiple times.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    contains : contains which allows common boundary points
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import Polygon
    >>> area1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    >>> area2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> area3 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])

    ``area1`` and ``area2`` have a common border:

    >>> contains(area1, area2)
    True
    >>> contains_properly(area1, area2)
    False

    ``area3`` is completely inside ``area1`` with no common border:

    >>> contains(area1, area3)
    True
    >>> contains_properly(area1, area3)
    True
    """
    return lib.contains_properly(a, b, **kwargs)


@multithreading_enabled
def covered_by(a, b, **kwargs):
    """Returns True if no point in geometry A is outside geometry B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    covers : ``covered_by(A, B) == covers(B, A)``
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> line = LineString([(0, 0), (1, 1)])
    >>> covered_by(Point(0, 0), line)
    True
    >>> covered_by(Point(0.5, 0.5), line)
    True
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> covered_by(Point(0, 0), area)
    True
    >>> covered_by(line, area)
    True
    >>> covered_by(LineString([(0, 0), (2, 2)]), area)
    False
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> covered_by(Point(1, 1), polygon_with_hole)
    True
    >>> covered_by(Point(2, 2), polygon_with_hole)
    True
    >>> covered_by(LineString([(1, 1), (5, 5)]), polygon_with_hole)
    False
    >>> covered_by(area, area)
    True
    >>> covered_by(None, area)
    False
    """
    return lib.covered_by(a, b, **kwargs)


@multithreading_enabled
def covers(a, b, **kwargs):
    """Returns True if no point in geometry B is outside geometry A.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    covered_by : ``covers(A, B) == covered_by(B, A)``
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> line = LineString([(0, 0), (1, 1)])
    >>> covers(line, Point(0, 0))
    True
    >>> covers(line, Point(0.5, 0.5))
    True
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> covers(area, Point(0, 0))
    True
    >>> covers(area, line)
    True
    >>> covers(area, LineString([(0, 0), (2, 2)]))
    False
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> covers(polygon_with_hole, Point(1, 1))
    True
    >>> covers(polygon_with_hole, Point(2, 2))
    True
    >>> covers(polygon_with_hole, LineString([(1, 1), (5, 5)]))
    False
    >>> covers(area, area)
    True
    >>> covers(area, None)
    False
    """
    return lib.covers(a, b, **kwargs)


@multithreading_enabled
def disjoint(a, b, **kwargs):
    """Returns True if A and B do not share any point in space.

    Disjoint implies that overlaps, touches, within, and intersects are False.
    Note missing (None) values are never disjoint.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    intersects : ``disjoint(A, B) == ~intersects(A, B)``
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, Point
    >>> line = LineString([(0, 0), (1, 1)])
    >>> disjoint(line, Point(0, 0))
    False
    >>> disjoint(line, Point(0, 1))
    True
    >>> disjoint(line, LineString([(0, 2), (2, 0)]))
    False
    >>> empty = GeometryCollection()
    >>> disjoint(line, empty)
    True
    >>> disjoint(empty, empty)
    True
    >>> disjoint(empty, None)
    False
    >>> disjoint(None, None)
    False
    """
    return lib.disjoint(a, b, **kwargs)


@multithreading_enabled
def equals(a, b, **kwargs):
    """Returns True if A and B are spatially equal.

    If A is within B and B is within A, A and B are considered equal. The
    ordering of points can be different.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    equals_exact : Check if A and B are structurally equal given a specified
        tolerance.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, Polygon
    >>> line = LineString([(0, 0), (5, 5), (10, 10)])
    >>> equals(line, LineString([(0, 0), (10, 10)]))
    True
    >>> equals(Polygon(), GeometryCollection())
    True
    >>> equals(None, None)
    False
    """
    return lib.equals(a, b, **kwargs)


@multithreading_enabled
def intersects(a, b, **kwargs):
    """Returns True if A and B share any portion of space.

    Intersects implies that overlaps, touches, covers, or within are True.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    disjoint : ``intersects(A, B) == ~disjoint(A, B)``
    prepare : improve performance by preparing ``a`` (the first argument)
    intersects_xy : variant for checking against a Point with x, y coordinates

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> line = LineString([(0, 0), (1, 1)])
    >>> intersects(line, Point(0, 0))
    True
    >>> intersects(line, Point(0, 1))
    False
    >>> intersects(line, LineString([(0, 2), (2, 0)]))
    True
    >>> intersects(None, None)
    False
    """
    return lib.intersects(a, b, **kwargs)


@multithreading_enabled
def overlaps(a, b, **kwargs):
    """Returns True if A and B spatially overlap.

    A and B overlap if they have some but not all points in common, have the
    same dimension, and the intersection of the interiors of the two geometries
    has the same dimension as the geometries themselves.  That is, only polyons
    can overlap other polygons and only lines can overlap other lines.

    If either A or B are None, the output is always False.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> poly = Polygon([(0, 0), (0, 4), (4, 4), (4, 0), (0, 0)])
    >>> # A and B share all points (are spatially equal):
    >>> overlaps(poly, poly)
    False
    >>> # A contains B; all points of B are within A:
    >>> overlaps(poly, Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]))
    False
    >>> # A partially overlaps with B:
    >>> overlaps(poly, Polygon([(2, 2), (2, 6), (6, 6), (6, 2), (2, 2)]))
    True
    >>> line = LineString([(2, 2), (6, 6)])
    >>> # A and B are different dimensions; they cannot overlap:
    >>> overlaps(poly, line)
    False
    >>> overlaps(poly, Point(2, 2))
    False
    >>> # A and B share some but not all points:
    >>> overlaps(line, LineString([(0, 0), (4, 4)]))
    True
    >>> # A and B intersect only at a point (lower dimension); they do not overlap
    >>> overlaps(line, LineString([(6, 0), (0, 6)]))
    False
    >>> overlaps(poly, None)
    False
    >>> overlaps(None, None)
    False
    """
    return lib.overlaps(a, b, **kwargs)


@multithreading_enabled
def touches(a, b, **kwargs):
    """Returns True if the only points shared between A and B are on the
    boundary of A and B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> line = LineString([(0, 2), (2, 0)])
    >>> touches(line, Point(0, 2))
    True
    >>> touches(line, Point(1, 1))
    False
    >>> touches(line, LineString([(0, 0), (1, 1)]))
    True
    >>> touches(line, LineString([(0, 0), (2, 2)]))
    False
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> touches(area, Point(0.5, 0))
    True
    >>> touches(area, Point(0.5, 0.5))
    False
    >>> touches(area, line)
    True
    >>> touches(area, Polygon([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1)]))
    True
    """
    return lib.touches(a, b, **kwargs)


@multithreading_enabled
def within(a, b, **kwargs):
    """Returns True if geometry A is completely inside geometry B.

    A is within B if no points of A lie in the exterior of B and at least one
    point of the interior of A lies in the interior of B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    contains : ``within(A, B) == contains(B, A)``
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> line = LineString([(0, 0), (1, 1)])
    >>> within(Point(0, 0), line)
    False
    >>> within(Point(0.5, 0.5), line)
    True
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> within(Point(0, 0), area)
    False
    >>> within(line, area)
    True
    >>> within(LineString([(0, 0), (2, 2)]), area)
    False
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> within(Point(1, 1), polygon_with_hole)
    True
    >>> within(Point(2, 2), polygon_with_hole)
    False
    >>> within(LineString([(1, 1), (5, 5)]), polygon_with_hole)
    False
    >>> within(area, area)
    True
    >>> within(None, area)
    False
    """
    return lib.within(a, b, **kwargs)


@multithreading_enabled
def equals_exact(a, b, tolerance=0.0, **kwargs):
    """Returns True if A and B are structurally equal.

    This method uses exact coordinate equality, which requires coordinates
    to be equal (within specified tolerance) and and in the same order for all
    components of a geometry. This is in contrast with the ``equals`` function
    which uses spatial (topological) equality.

    Parameters
    ----------
    a, b : Geometry or array_like
    tolerance : float or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    equals : Check if A and B are spatially equal.

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> point1 = Point(50, 50)
    >>> point2 = Point(50.1, 50.1)
    >>> equals_exact(point1, point2)
    False
    >>> equals_exact(point1, point2, tolerance=0.2)
    True
    >>> equals_exact(point1, None, tolerance=0.2)
    False

    Difference between structucal and spatial equality:

    >>> polygon1 = Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])
    >>> polygon2 = Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])
    >>> equals_exact(polygon1, polygon2)
    False
    >>> equals(polygon1, polygon2)
    True
    """
    return lib.equals_exact(a, b, tolerance, **kwargs)


def relate(a, b, **kwargs):
    """
    Returns a string representation of the DE-9IM intersection matrix.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> point = Point(0, 0)
    >>> line = LineString([(0, 0), (1, 1)])
    >>> relate(point, line)
    'F0FFFF102'
    """
    return lib.relate(a, b, **kwargs)


@multithreading_enabled
def relate_pattern(a, b, pattern, **kwargs):
    """
    Returns True if the DE-9IM string code for the relationship between
    the geometries satisfies the pattern, else False.

    This function compares the DE-9IM code string for two geometries
    against a specified pattern. If the string matches the pattern then
    ``True`` is returned, otherwise ``False``. The pattern specified can
    be an exact match (``0``, ``1`` or ``2``), a boolean match
    (uppercase ``T`` or ``F``), or a wildcard (``*``). For example,
    the pattern for the ``within`` predicate is ``'T*F**F***'``.

    Parameters
    ----------
    a, b : Geometry or array_like
    pattern : string
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> point = Point(0.5, 0.5)
    >>> square = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> relate(point, square)
    '0FFFFF212'
    >>> relate_pattern(point, square, "T*F**F***")
    True
    """
    return lib.relate_pattern(a, b, pattern, **kwargs)


@multithreading_enabled
@requires_geos("3.10.0")
def dwithin(a, b, distance, **kwargs):
    """
    Returns True if the geometries are within a given distance.

    Using this function is more efficient than computing the distance and
    comparing the result.

    Parameters
    ----------
    a, b : Geometry or array_like
    distance : float
        Negative distances always return False.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    distance : compute the actual distance between A and B
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> from shapely import Point
    >>> point = Point(0.5, 0.5)
    >>> dwithin(point, Point(2, 0.5), 2)
    True
    >>> dwithin(point, Point(2, 0.5), [2, 1.5, 1]).tolist()
    [True, True, False]
    >>> dwithin(point, Point(0.5, 0.5), 0)
    True
    >>> dwithin(point, None, 100)
    False
    """
    return lib.dwithin(a, b, distance, **kwargs)


@multithreading_enabled
def contains_xy(geom, x, y=None, **kwargs):
    """
    Returns True if the Point (x, y) is completely inside geometry A.

    This is a special-case (and faster) variant of the `contains` function
    which avoids having to create a Point object if you start from x/y
    coordinates.

    Note that in the case of points, the `contains_properly` predicate is
    equivalent to `contains`.

    See the docstring of `contains` for more details about the predicate.

    Parameters
    ----------
    geom : Geometry or array_like
    x, y : float or array_like
        Coordinates as separate x and y arrays, or a single array of
        coordinate x, y tuples.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    contains : variant taking two geometries as input

    Notes
    -----
    If you compare a small number of polygons or lines with many points,
    it can be beneficial to prepare the geometries in advance using
    :func:`shapely.prepare`.

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> contains(area, Point(0.5, 0.5))
    True
    >>> contains_xy(area, 0.5, 0.5)
    True
    """
    if y is None:
        coords = np.asarray(x)
        x, y = coords[:, 0], coords[:, 1]
    return lib.contains_xy(geom, x, y, **kwargs)


@multithreading_enabled
def intersects_xy(geom, x, y=None, **kwargs):
    """
    Returns True if A and the Point (x, y) share any portion of space.

    This is a special-case (and faster) variant of the `intersects` function
    which avoids having to create a Point object if you start from x/y
    coordinates.

    See the docstring of `intersects` for more details about the predicate.

    Parameters
    ----------
    geom : Geometry or array_like
    x, y : float or array_like
        Coordinates as separate x and y arrays, or a single array of
        coordinate x, y tuples.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    intersects : variant taking two geometries as input

    Notes
    -----
    If you compare a single or few geometries with many points, it can be
    beneficial to prepare the geometries in advance using
    :func:`shapely.prepare`.

    The `touches` predicate can be determined with this function by getting
    the boundary of the geometries: ``intersects_xy(boundary(geom), x, y)``.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> line = LineString([(0, 0), (1, 1)])
    >>> intersects(line, Point(0, 0))
    True
    >>> intersects_xy(line, 0, 0)
    True
    """
    if y is None:
        coords = np.asarray(x)
        x, y = coords[:, 0], coords[:, 1]
    return lib.intersects_xy(geom, x, y, **kwargs)
