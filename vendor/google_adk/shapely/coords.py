"""Coordinate sequence utilities
"""
from array import array


class CoordinateSequence:
    """
    Iterative access to coordinate tuples from the parent geometry's coordinate
    sequence.

    Example:

      >>> from shapely.wkt import loads
      >>> g = loads('POINT (0.0 0.0)')
      >>> list(g.coords)
      [(0.0, 0.0)]

    """

    def __init__(self, coords):
        self._coords = coords

    def __len__(self):
        return self._coords.shape[0]

    def __iter__(self):
        for i in range(self.__len__()):
            yield tuple(self._coords[i].tolist())

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return tuple(self._coords[i].tolist())
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(tuple(self._coords[i].tolist()))
            return res
        else:
            raise TypeError("key must be an index or slice")

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        elif copy is True:
            return self._coords.copy()
        else:
            return self._coords

    @property
    def xy(self):
        """X and Y arrays"""
        m = self.__len__()
        x = array("d")
        y = array("d")
        for i in range(m):
            xy = self._coords[i].tolist()
            x.append(xy[0])
            y.append(xy[1])
        return x, y
