"""Multi-part collections of geometries
"""

import shapely
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry


class GeometryCollection(BaseMultipartGeometry):
    """
    A collection of one or more geometries that may contain more than one type
    of geometry.

    Parameters
    ----------
    geoms : list
        A list of shapely geometry instances, which may be of varying
        geometry types.

    Attributes
    ----------
    geoms : sequence
        A sequence of Shapely geometry instances

    Examples
    --------
    Create a GeometryCollection with a Point and a LineString

    >>> from shapely import LineString, Point
    >>> p = Point(51, -1)
    >>> l = LineString([(52, -1), (49, 2)])
    >>> gc = GeometryCollection([p, l])
    """

    __slots__ = []

    def __new__(self, geoms=None):
        if not geoms:
            # TODO better empty constructor
            return shapely.from_wkt("GEOMETRYCOLLECTION EMPTY")
        if isinstance(geoms, BaseGeometry):
            # TODO(shapely-2.0) do we actually want to split Multi-part geometries?
            # this is needed for the split() tests
            if hasattr(geoms, "geoms"):
                geoms = geoms.geoms
            else:
                geoms = [geoms]

        return shapely.geometrycollections(geoms)

    @property
    def __geo_interface__(self):
        geometries = []
        for geom in self.geoms:
            geometries.append(geom.__geo_interface__)
        return dict(type="GeometryCollection", geometries=geometries)


shapely.lib.registry[7] = GeometryCollection
