import numpy as np

from shapely import GeometryType, lib
from shapely.decorators import multithreading_enabled, requires_geos
from shapely.errors import UnsupportedGEOSVersionError

__all__ = [
    "difference",
    "intersection",
    "intersection_all",
    "symmetric_difference",
    "symmetric_difference_all",
    "unary_union",
    "union",
    "union_all",
    "coverage_union",
    "coverage_union_all",
]


@multithreading_enabled
def difference(a, b, grid_size=None, **kwargs):
    """Returns the part of geometry A that does not intersect with geometry B.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional
        Precision grid size; requires GEOS >= 3.9.0.  Will use the highest
        precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    set_precision

    Examples
    --------
    >>> from shapely import box, LineString, normalize, Polygon
    >>> line = LineString([(0, 0), (2, 2)])
    >>> difference(line, LineString([(1, 1), (3, 3)]))
    <LINESTRING (0 0, 1 1)>
    >>> difference(line, LineString())
    <LINESTRING (0 0, 2 2)>
    >>> difference(line, None) is None
    True
    >>> box1 = box(0, 0, 2, 2)
    >>> box2 = box(1, 1, 3, 3)
    >>> normalize(difference(box1, box2))
    <POLYGON ((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0))>
    >>> box1 = box(0.1, 0.2, 2.1, 2.1)
    >>> difference(box1, box2, grid_size=1)
    <POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))>
    """

    if grid_size is not None:
        if lib.geos_version < (3, 9, 0):
            raise UnsupportedGEOSVersionError(
                "grid_size parameter requires GEOS >= 3.9.0"
            )

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.difference_prec(a, b, grid_size, **kwargs)

    return lib.difference(a, b, **kwargs)


@multithreading_enabled
def intersection(a, b, grid_size=None, **kwargs):
    """Returns the geometry that is shared between input geometries.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional
        Precision grid size; requires GEOS >= 3.9.0.  Will use the highest
        precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    intersection_all
    set_precision

    Examples
    --------
    >>> from shapely import box, LineString, normalize, Polygon
    >>> line = LineString([(0, 0), (2, 2)])
    >>> intersection(line, LineString([(1, 1), (3, 3)]))
    <LINESTRING (1 1, 2 2)>
    >>> box1 = box(0, 0, 2, 2)
    >>> box2 = box(1, 1, 3, 3)
    >>> normalize(intersection(box1, box2))
    <POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))>
    >>> box1 = box(0.1, 0.2, 2.1, 2.1)
    >>> intersection(box1, box2, grid_size=1)
    <POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))>
    """

    if grid_size is not None:
        if lib.geos_version < (3, 9, 0):
            raise UnsupportedGEOSVersionError(
                "grid_size parameter requires GEOS >= 3.9.0"
            )

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.intersection_prec(a, b, grid_size, **kwargs)

    return lib.intersection(a, b, **kwargs)


@multithreading_enabled
def intersection_all(geometries, axis=None, **kwargs):
    """Returns the intersection of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, an empty GeometryCollection is
    returned.

    Parameters
    ----------
    geometries : array_like
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    intersection

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(1, 1), (3, 3)])
    >>> intersection_all([line1, line2])
    <LINESTRING (1 1, 2 2)>
    >>> intersection_all([[line1, line2, None]], axis=1).tolist()
    [<LINESTRING (1 1, 2 2)>]
    >>> intersection_all([line1, None])
    <LINESTRING (0 0, 2 2)>
    """
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)

    return lib.intersection_all(geometries, **kwargs)


@multithreading_enabled
def symmetric_difference(a, b, grid_size=None, **kwargs):
    """Returns the geometry that represents the portions of input geometries
    that do not intersect.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional
        Precision grid size; requires GEOS >= 3.9.0.  Will use the highest
        precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    symmetric_difference_all
    set_precision

    Examples
    --------
    >>> from shapely import box, LineString, normalize
    >>> line = LineString([(0, 0), (2, 2)])
    >>> symmetric_difference(line, LineString([(1, 1), (3, 3)]))
    <MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> box1 = box(0, 0, 2, 2)
    >>> box2 = box(1, 1, 3, 3)
    >>> normalize(symmetric_difference(box1, box2))
    <MULTIPOLYGON (((1 2, 1 3, 3 3, 3 1, 2 1, 2 2, 1 2)), ((0 0, 0 2, 1 2, 1 1, ...>
    >>> box1 = box(0.1, 0.2, 2.1, 2.1)
    >>> symmetric_difference(box1, box2, grid_size=1)
    <MULTIPOLYGON (((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0)), ((2 2, 1 2, 1 3, 3 3, ...>
    """

    if grid_size is not None:
        if lib.geos_version < (3, 9, 0):
            raise UnsupportedGEOSVersionError(
                "grid_size parameter requires GEOS >= 3.9.0"
            )

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.symmetric_difference_prec(a, b, grid_size, **kwargs)

    return lib.symmetric_difference(a, b, **kwargs)


@multithreading_enabled
def symmetric_difference_all(geometries, axis=None, **kwargs):
    """Returns the symmetric difference of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None an empty GeometryCollection is
    returned.

    Parameters
    ----------
    geometries : array_like
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    symmetric_difference

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(1, 1), (3, 3)])
    >>> symmetric_difference_all([line1, line2])
    <MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> symmetric_difference_all([[line1, line2, None]], axis=1).tolist()
    [<MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>]
    >>> symmetric_difference_all([line1, None])
    <LINESTRING (0 0, 2 2)>
    >>> symmetric_difference_all([None, None])
    <GEOMETRYCOLLECTION EMPTY>
    """
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)

    return lib.symmetric_difference_all(geometries, **kwargs)


@multithreading_enabled
def union(a, b, grid_size=None, **kwargs):
    """Merges geometries into one.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional
        Precision grid size; requires GEOS >= 3.9.0.  Will use the highest
        precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    union_all
    set_precision

    Examples
    --------
    >>> from shapely import box, LineString, normalize
    >>> line = LineString([(0, 0), (2, 2)])
    >>> union(line, LineString([(2, 2), (3, 3)]))
    <MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> union(line, None) is None
    True
    >>> box1 = box(0, 0, 2, 2)
    >>> box2 = box(1, 1, 3, 3)
    >>> normalize(union(box1, box2))
    <POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    >>> box1 = box(0.1, 0.2, 2.1, 2.1)
    >>> union(box1, box2, grid_size=1)
    <POLYGON ((2 0, 0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0))>
    """

    if grid_size is not None:
        if lib.geos_version < (3, 9, 0):
            raise UnsupportedGEOSVersionError(
                "grid_size parameter requires GEOS >= 3.9.0"
            )

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.union_prec(a, b, grid_size, **kwargs)

    return lib.union(a, b, **kwargs)


@multithreading_enabled
def union_all(geometries, grid_size=None, axis=None, **kwargs):
    """Returns the union of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None an empty GeometryCollection is
    returned.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    `unary_union` is an alias of `union_all`.

    Parameters
    ----------
    geometries : array_like
    grid_size : float, optional
        Precision grid size; requires GEOS >= 3.9.0.  Will use the highest
        precision of the inputs by default.
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    union
    set_precision

    Examples
    --------
    >>> from shapely import box, LineString, normalize, Point
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(2, 2), (3, 3)])
    >>> union_all([line1, line2])
    <MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> union_all([[line1, line2, None]], axis=1).tolist()
    [<MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>]
    >>> box1 = box(0, 0, 2, 2)
    >>> box2 = box(1, 1, 3, 3)
    >>> normalize(union_all([box1, box2]))
    <POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    >>> box1 = box(0.1, 0.2, 2.1, 2.1)
    >>> union_all([box1, box2], grid_size=1)
    <POLYGON ((2 0, 0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0))>
    >>> union_all([None, Point(0, 1)])
    <POINT (0 1)>
    >>> union_all([None, None])
    <GEOMETRYCOLLECTION EMPTY>
    >>> union_all([])
    <GEOMETRYCOLLECTION EMPTY>
    """
    # for union_all, GEOS provides an efficient route through first creating
    # GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)

    # create_collection acts on the inner axis
    collections = lib.create_collection(
        geometries, np.intc(GeometryType.GEOMETRYCOLLECTION)
    )

    if grid_size is not None:
        if lib.geos_version < (3, 9, 0):
            raise UnsupportedGEOSVersionError(
                "grid_size parameter requires GEOS >= 3.9.0"
            )

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.unary_union_prec(collections, grid_size, **kwargs)

    return lib.unary_union(collections, **kwargs)


unary_union = union_all


@requires_geos("3.8.0")
@multithreading_enabled
def coverage_union(a, b, **kwargs):
    """Merges multiple polygons into one. This is an optimized version of
    union which assumes the polygons to be non-overlapping.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    coverage_union_all

    Examples
    --------
    >>> from shapely import normalize, Polygon
    >>> polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> normalize(coverage_union(polygon, Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])))
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>

    Union with None returns same polygon
    >>> normalize(coverage_union(polygon, None))
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    return coverage_union_all([a, b], **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def coverage_union_all(geometries, axis=None, **kwargs):
    """Returns the union of multiple polygons of a geometry collection.
    This is an optimized version of union which assumes the polygons
    to be non-overlapping.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, an empty MultiPolygon is
    returned.

    Parameters
    ----------
    geometries : array_like
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    coverage_union

    Examples
    --------
    >>> from shapely import normalize, Polygon
    >>> polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> polygon_2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
    >>> normalize(coverage_union_all([polygon_1, polygon_2]))
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>
    >>> normalize(coverage_union_all([polygon_1, None]))
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    >>> normalize(coverage_union_all([None, None]))
    <MULTIPOLYGON EMPTY>
    """
    # coverage union in GEOS works over GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(
        geometries, np.intc(GeometryType.GEOMETRYCOLLECTION)
    )
    return lib.coverage_union(collections, **kwargs)
