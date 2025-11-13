"""Geometry classes and factories
"""

from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.geo import box, mapping, shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import LinearRing, Polygon

__all__ = [
    "box",
    "shape",
    "mapping",
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
    "LinearRing",
    "CAP_STYLE",
    "JOIN_STYLE",
]
