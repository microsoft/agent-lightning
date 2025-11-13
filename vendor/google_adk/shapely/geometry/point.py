"""Points and related utilities
"""
import numpy as np

import shapely
from shapely.errors import DimensionError
from shapely.geometry.base import BaseGeometry

__all__ = ["Point"]


class Point(BaseGeometry):
    """
    A geometry type that represents a single coordinate with
    x,y and possibly z values.

    A point is a zero-dimensional feature and has zero length and zero area.

    Parameters
    ----------
    args : float, or sequence of floats
        The coordinates can either be passed as a single parameter, or as
        individual float values using multiple parameters:

        1) 1 parameter: a sequence or array-like of with 2 or 3 values.
        2) 2 or 3 parameters (float): x, y, and possibly z.

    Attributes
    ----------
    x, y, z : float
        Coordinate values

    Examples
    --------
    Constructing the Point using separate parameters for x and y:

    >>> p = Point(1.0, -1.0)

    Constructing the Point using a list of x, y coordinates:

    >>> p = Point([1.0, -1.0])
    >>> print(p)
    POINT (1 -1)
    >>> p.y
    -1.0
    >>> p.x
    1.0
    """

    __slots__ = []

    def __new__(self, *args):
        if len(args) == 0:
            # empty geometry
            # TODO better constructor
            return shapely.from_wkt("POINT EMPTY")
        elif len(args) > 3:
            raise TypeError(f"Point() takes at most 3 arguments ({len(args)} given)")
        elif len(args) == 1:
            coords = args[0]
            if isinstance(coords, Point):
                return coords

            # Accept either (x, y) or [(x, y)]
            if not hasattr(coords, "__getitem__"):  # generators
                coords = list(coords)
            coords = np.asarray(coords).squeeze()
        else:
            # 2 or 3 args
            coords = np.array(args).squeeze()

        if coords.ndim > 1:
            raise ValueError(
                f"Point() takes only scalar or 1-size vector arguments, got {args}"
            )
        if not np.issubdtype(coords.dtype, np.number):
            coords = [float(c) for c in coords]
        geom = shapely.points(coords)
        if not isinstance(geom, Point):
            raise ValueError("Invalid values passed to Point constructor")
        return geom

    # Coordinate getters and setters

    @property
    def x(self):
        """Return x coordinate."""
        return float(shapely.get_x(self))

    @property
    def y(self):
        """Return y coordinate."""
        return float(shapely.get_y(self))

    @property
    def z(self):
        """Return z coordinate."""
        if not shapely.has_z(self):
            raise DimensionError("This point has no z coordinate.")
        # return shapely.get_z(self) -> get_z only supported for GEOS 3.7+
        return self.coords[0][2]

    @property
    def __geo_interface__(self):
        return {"type": "Point", "coordinates": self.coords[0]}

    def svg(self, scale_factor=1.0, fill_color=None, opacity=None):
        """Returns SVG circle element for the Point geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameter.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Default value is 0.6
        """
        if self.is_empty:
            return "<g />"
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        if opacity is None:
            opacity = 0.6
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="{4}" />'
        ).format(self, 3.0 * scale_factor, 1.0 * scale_factor, fill_color, opacity)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values

        Example:
          >>> x, y = Point(0, 0).xy
          >>> list(x)
          [0.0]
          >>> list(y)
          [0.0]
        """
        return self.coords.xy


shapely.lib.registry[0] = Point
