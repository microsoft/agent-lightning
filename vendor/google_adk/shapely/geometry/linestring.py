"""Line strings and related utilities
"""
import numpy as np

import shapely
from shapely.geometry.base import BaseGeometry, JOIN_STYLE
from shapely.geometry.point import Point

__all__ = ["LineString"]


class LineString(BaseGeometry):
    """
    A geometry type composed of one or more line segments.

    A LineString is a one-dimensional feature and has a non-zero length but
    zero area. It may approximate a curve and need not be straight. Unlike a
    LinearRing, a LineString is not closed.

    Parameters
    ----------
    coordinates : sequence
        A sequence of (x, y, [,z]) numeric coordinate pairs or triples, or
        an array-like with shape (N, 2) or (N, 3).
        Also can be a sequence of Point objects.

    Examples
    --------
    Create a LineString with two segments

    >>> a = LineString([[0, 0], [1, 0], [1, 1]])
    >>> a.length
    2.0
    """

    __slots__ = []

    def __new__(self, coordinates=None):
        if coordinates is None:
            # empty geometry
            # TODO better constructor
            return shapely.from_wkt("LINESTRING EMPTY")
        elif isinstance(coordinates, LineString):
            if type(coordinates) == LineString:
                # return original objects since geometries are immutable
                return coordinates
            else:
                # LinearRing
                # TODO convert LinearRing to LineString more directly
                coordinates = coordinates.coords
        else:
            if hasattr(coordinates, "__array__"):
                coordinates = np.asarray(coordinates)
            if isinstance(coordinates, np.ndarray) and np.issubdtype(
                coordinates.dtype, np.number
            ):
                pass
            else:
                # check coordinates on points
                def _coords(o):
                    if isinstance(o, Point):
                        return o.coords[0]
                    else:
                        return [float(c) for c in o]

                coordinates = [_coords(o) for o in coordinates]

        if len(coordinates) == 0:
            # empty geometry
            # TODO better constructor + should shapely.linestrings handle this?
            return shapely.from_wkt("LINESTRING EMPTY")

        geom = shapely.linestrings(coordinates)
        if not isinstance(geom, LineString):
            raise ValueError("Invalid values passed to LineString constructor")
        return geom

    @property
    def __geo_interface__(self):
        return {"type": "LineString", "coordinates": tuple(self.coords)}

    def svg(self, scale_factor=1.0, stroke_color=None, opacity=None):
        """Returns SVG polyline element for the LineString geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        stroke_color : str, optional
            Hex string for stroke color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Default value is 0.8
        """
        if self.is_empty:
            return "<g />"
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        if opacity is None:
            opacity = 0.8
        pnt_format = " ".join(["{},{}".format(*c) for c in self.coords])
        return (
            '<polyline fill="none" stroke="{2}" stroke-width="{1}" '
            'points="{0}" opacity="{3}" />'
        ).format(pnt_format, 2.0 * scale_factor, stroke_color, opacity)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values

        Example:

          >>> x, y = LineString([(0, 0), (1, 1)]).xy
          >>> list(x)
          [0.0, 1.0]
          >>> list(y)
          [0.0, 1.0]
        """
        return self.coords.xy

    def offset_curve(
        self,
        distance,
        quad_segs=16,
        join_style=JOIN_STYLE.round,
        mitre_limit=5.0,
    ):
        """Returns a LineString or MultiLineString geometry at a distance from
        the object on its right or its left side.

        The side is determined by the sign of the `distance` parameter
        (negative for right side offset, positive for left side offset). The
        resolution of the buffer around each vertex of the object increases
        by increasing the `quad_segs` keyword parameter.

        The join style is for outside corners between line segments. Accepted
        values are JOIN_STYLE.round (1), JOIN_STYLE.mitre (2), and
        JOIN_STYLE.bevel (3).

        The mitre ratio limit is used for very sharp corners. It is the ratio
        of the distance from the corner to the end of the mitred offset corner.
        When two line segments meet at a sharp angle, a miter join will extend
        far beyond the original geometry. To prevent unreasonable geometry, the
        mitre limit allows controlling the maximum length of the join corner.
        Corners with a ratio which exceed the limit will be beveled.

        Note: the behaviour regarding orientation of the resulting line
        depends on the GEOS version. With GEOS < 3.11, the line retains the
        same direction for a left offset (positive distance) or has reverse
        direction for a right offset (negative distance), and this behaviour
        was documented as such in previous Shapely versions. Starting with
        GEOS 3.11, the function tries to preserve the orientation of the
        original line.
        """
        if mitre_limit == 0.0:
            raise ValueError("Cannot compute offset from zero-length line segment")
        elif not np.isfinite(distance):
            raise ValueError("offset_curve distance must be finite")
        return shapely.offset_curve(self, distance, quad_segs, join_style, mitre_limit)

    def parallel_offset(
        self,
        distance,
        side="right",
        resolution=16,
        join_style=JOIN_STYLE.round,
        mitre_limit=5.0,
    ):
        """
        Alternative method to :meth:`offset_curve` method.

        Older alternative method to the :meth:`offset_curve` method, but uses
        ``resolution`` instead of ``quad_segs`` and a ``side`` keyword
        ('left' or 'right') instead of sign of the distance. This method is
        kept for backwards compatibility for now, but is is recommended to
        use :meth:`offset_curve` instead.
        """
        if side == "right":
            distance *= -1
        return self.offset_curve(
            distance,
            quad_segs=resolution,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )


shapely.lib.registry[1] = LineString
