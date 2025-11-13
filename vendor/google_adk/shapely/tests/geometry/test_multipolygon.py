import numpy as np
import pytest

from shapely import MultiPolygon, Polygon
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase


class TestMultiPolygon(MultiGeometryTestCase):
    def test_multipolygon(self):
        # From coordinate tuples
        coords = [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
        geom = MultiPolygon(coords)
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
            ]
        ]

        # Or without holes
        coords2 = [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),)]
        geom = MultiPolygon(coords2)
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
            ]
        ]

        # Or from polygons
        p = Polygon(
            ((0, 0), (0, 1), (1, 1), (1, 0)),
            [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
        )
        geom = MultiPolygon([p])
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
            ]
        ]

        # Or from another multi-polygon
        geom2 = MultiPolygon(geom)
        assert len(geom2.geoms) == 1
        assert dump_coords(geom2) == [
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
            ]
        ]

        # Sub-geometry Access
        assert isinstance(geom.geoms[0], Polygon)
        assert dump_coords(geom.geoms[0]) == [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
            [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)],
        ]
        with pytest.raises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface
        assert geom.__geo_interface__ == {
            "type": "MultiPolygon",
            "coordinates": [
                (
                    ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                    ((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)),
                )
            ],
        }

    def test_subgeom_access(self):
        poly0 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
        poly1 = Polygon([(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)])
        self.subgeom_access_test(MultiPolygon, [poly0, poly1])


def test_fail_list_of_multipolygons():
    """A list of multipolygons is not a valid multipolygon ctor argument"""
    multi = MultiPolygon(
        [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
    )
    with pytest.raises(ValueError):
        MultiPolygon([multi])


def test_numpy_object_array():
    geom = MultiPolygon(
        [
            (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))],
            )
        ]
    )
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom
