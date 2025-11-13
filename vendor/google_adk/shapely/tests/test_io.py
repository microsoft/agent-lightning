import json
import pickle
import struct
import warnings

import numpy as np
import pytest

import shapely
from shapely import GeometryCollection, LineString, Point, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty_point, empty_point_z, point, point_z

# fmt: off
POINT11_WKB = b"\x01\x01\x00\x00\x00" + struct.pack("<2d", 1.0, 1.0)
NAN = struct.pack("<d", float("nan"))
POINT_NAN_WKB = b'\x01\x01\x00\x00\x00' + (NAN * 2)
POINTZ_NAN_WKB = b'\x01\x01\x00\x00\x80' + (NAN * 3)
MULTIPOINT_NAN_WKB = b'\x01\x04\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00' + (NAN * 2)
MULTIPOINTZ_NAN_WKB = b'\x01\x04\x00\x00\x80\x01\x00\x00\x00\x01\x01\x00\x00\x80' + (NAN * 3)
GEOMETRYCOLLECTION_NAN_WKB = b'\x01\x07\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00' + (NAN * 2)
GEOMETRYCOLLECTIONZ_NAN_WKB = b'\x01\x07\x00\x00\x80\x01\x00\x00\x00\x01\x01\x00\x00\x80' + (NAN * 3)
NESTED_COLLECTION_NAN_WKB = b'\x01\x07\x00\x00\x00\x01\x00\x00\x00\x01\x04\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00' + (NAN * 2)
NESTED_COLLECTIONZ_NAN_WKB = b'\x01\x07\x00\x00\x80\x01\x00\x00\x00\x01\x04\x00\x00\x80\x01\x00\x00\x00\x01\x01\x00\x00\x80' + (NAN * 3)
INVALID_WKB = "01030000000100000002000000507daec600b1354100de02498e5e3d41306ea321fcb03541a011a53d905e3d41"
# fmt: on


GEOJSON_GEOMETRY = json.dumps({"type": "Point", "coordinates": [125.6, 10.1]}, indent=4)
GEOJSON_FEATURE = json.dumps(
    {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [125.6, 10.1]},
        "properties": {"name": "Dinagat Islands"},
    },
    indent=4,
)
GEOJSON_FEATURECOLECTION = json.dumps(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [102.0, 0.6]},
                "properties": {"prop0": "value0"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [102.0, 0.0],
                        [103.0, 1.0],
                        [104.0, 0.0],
                        [105.0, 1.0],
                    ],
                },
                "properties": {"prop1": 0.0, "prop0": "value0"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [100.0, 0.0],
                            [101.0, 0.0],
                            [101.0, 1.0],
                            [100.0, 1.0],
                            [100.0, 0.0],
                        ]
                    ],
                },
                "properties": {"prop1": {"this": "that"}, "prop0": "value0"},
            },
        ],
    },
    indent=4,
)

GEOJSON_GEOMETRY_EXPECTED = shapely.points(125.6, 10.1)
GEOJSON_COLLECTION_EXPECTED = [
    shapely.points([102.0, 0.6]),
    shapely.linestrings([[102.0, 0.0], [103.0, 1.0], [104.0, 0.0], [105.0, 1.0]]),
    shapely.polygons(
        [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]
    ),
]


def test_from_wkt():
    expected = shapely.points(1, 1)
    actual = shapely.from_wkt("POINT (1 1)")
    assert_geometries_equal(actual, expected)
    # also accept bytes
    actual = shapely.from_wkt(b"POINT (1 1)")
    assert_geometries_equal(actual, expected)


def test_from_wkt_none():
    # None propagates
    assert shapely.from_wkt(None) is None


def test_from_wkt_exceptions():
    with pytest.raises(TypeError, match="Expected bytes or string, got int"):
        shapely.from_wkt(1)

    with pytest.raises(
        shapely.GEOSException, match="Expected word but encountered end of stream"
    ):
        shapely.from_wkt("")

    with pytest.raises(shapely.GEOSException, match="Unknown type: 'NOT'"):
        shapely.from_wkt("NOT A WKT STRING")


def test_from_wkt_warn_on_invalid():
    with pytest.warns(Warning, match="Invalid WKT"):
        shapely.from_wkt("", on_invalid="warn")

    with pytest.warns(Warning, match="Invalid WKT"):
        shapely.from_wkt("NOT A WKT STRING", on_invalid="warn")


def test_from_wkb_ignore_on_invalid():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shapely.from_wkt("", on_invalid="ignore")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shapely.from_wkt("NOT A WKT STRING", on_invalid="ignore")


def test_from_wkt_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        shapely.from_wkt(b"\x01\x01\x00\x00\x00\x00", on_invalid="unsupported_option")


@pytest.mark.parametrize("geom", all_types)
def test_from_wkt_all_types(geom):
    wkt = shapely.to_wkt(geom)
    actual = shapely.from_wkt(wkt)
    assert_geometries_equal(actual, geom)


@pytest.mark.parametrize(
    "wkt",
    ("POINT EMPTY", "LINESTRING EMPTY", "POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY"),
)
def test_from_wkt_empty(wkt):
    geom = shapely.from_wkt(wkt)
    assert shapely.is_geometry(geom).all()
    assert shapely.is_empty(geom).all()
    assert shapely.to_wkt(geom) == wkt


# WKT from https://github.com/libgeos/geos/blob/main/tests/unit/io/WKBReaderTest.cpp
@pytest.mark.parametrize(
    "wkt",
    (
        "CIRCULARSTRING(1 3,2 4,3 1)",
        "COMPOUNDCURVE(CIRCULARSTRING(1 3,2 4,3 1),(3 1,0 0))",
        "CURVEPOLYGON(COMPOUNDCURVE(CIRCULARSTRING(0 0,2 0,2 1,2 3,4 3),(4 3,4 5,1 4,0 0)),CIRCULARSTRING(1.7 1,1.4 0.4,1.6 0.4,1.6 0.5,1.7 1))",  # noqa: E501
        "MULTICURVE((0 0,5 5),COMPOUNDCURVE((-1 -1,0 0),CIRCULARSTRING(0 0,1 1,2 0)),CIRCULARSTRING(4 0,4 4,8 4))",  # noqa: E501
        "MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0),(1 1,3 3,3 1,1 1)),((10 10,14 12,11 10,10 10),(11 11,11.5 11,11 11.5,11 11)))",  # noqa: E501
    ),
)
def test_from_wkt_nonlinear_unsupported(wkt):
    if shapely.geos_version >= (3, 13, 0):
        with pytest.raises(
            NotImplementedError,
            match="Nonlinear geometry types are not currently supported",
        ):
            shapely.from_wkt(wkt)

    else:
        # prior to GEOS 3.13 nonlinear types were rejected by GEOS on read from WKT
        with pytest.raises(shapely.errors.GEOSException, match="Unknown type"):
            shapely.from_wkt(wkt)


def test_from_wkb():
    expected = shapely.points(1, 1)
    actual = shapely.from_wkb(POINT11_WKB)
    assert_geometries_equal(actual, expected)


def test_from_wkb_hex():
    # HEX form
    expected = shapely.points(1, 1)
    actual = shapely.from_wkb("0101000000000000000000F03F000000000000F03F")
    assert_geometries_equal(actual, expected)
    actual = shapely.from_wkb(b"0101000000000000000000F03F000000000000F03F")
    assert_geometries_equal(actual, expected)


def test_from_wkb_none():
    # None propagates
    assert shapely.from_wkb(None) is None


def test_from_wkb_exceptions():
    with pytest.raises(TypeError, match="Expected bytes or string, got int"):
        shapely.from_wkb(1)

    # invalid WKB
    with pytest.raises(
        shapely.GEOSException,
        match=(
            "Unexpected EOF parsing WKB|"
            "ParseException: Input buffer is smaller than requested object size"
        ),
    ):
        result = shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00")
        assert result is None

    # invalid ring in WKB
    with pytest.raises(
        shapely.GEOSException,
        match="Points of LinearRing do not form a closed linestring",
    ):
        result = shapely.from_wkb(INVALID_WKB)
        assert result is None


def test_from_wkb_warn_on_invalid_warn():
    # invalid WKB
    with pytest.warns(Warning, match="Invalid WKB"):
        result = shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="warn")
        assert result is None

    # invalid ring in WKB
    with pytest.warns(Warning, match="Invalid WKB"):
        result = shapely.from_wkb(INVALID_WKB, on_invalid="warn")
        assert result is None


def test_from_wkb_ignore_on_invalid_ignore():
    # invalid WKB
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning
        result = shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="ignore")
        assert result is None

    # invalid ring in WKB
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning
        result = shapely.from_wkb(INVALID_WKB, on_invalid="ignore")
        assert result is None


def test_from_wkb_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="unsupported_option")


@pytest.mark.parametrize("geom", all_types)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types(geom, use_hex, byte_order):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in WKB")
    wkb = shapely.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = shapely.from_wkb(wkb)
    assert_geometries_equal(actual, geom)


@pytest.mark.parametrize(
    "geom",
    (Point(), LineString(), Polygon(), GeometryCollection()),
)
def test_from_wkb_empty(geom):
    wkb = shapely.to_wkb(geom)
    geom = shapely.from_wkb(wkb)
    assert shapely.is_geometry(geom).all()
    assert shapely.is_empty(geom).all()
    assert shapely.to_wkb(geom) == wkb


# WKB from https://github.com/libgeos/geos/blob/main/tests/unit/io/WKBReaderTest.cpp
@pytest.mark.parametrize(
    "wkb",
    (
        # "CIRCULARSTRING(1 3,2 4,3 1)",
        "010800000003000000000000000000F03F0000000000000840000000000000004000000000000010400000000000000840000000000000F03F",
        # "COMPOUNDCURVE(CIRCULARSTRING(1 3,2 4,3 1),(3 1,0 0))",
        "01090000200E16000002000000010800000003000000000000000000F03F0000000000000840000000000000004000000000000010400000000000000840000000000000F03F0102000000020000000000000000000840000000000000F03F00000000000000000000000000000000",
        # "CURVEPOLYGON(COMPOUNDCURVE(CIRCULARSTRING(0 0,2 0,2 1,2 3,4 3),(4 3,4 5,1 4,0 0)),CIRCULARSTRING(1.7 1,1.4 0.4,1.6 0.4,1.6 0.5,1.7 1))",  # noqa: E501
        "010A0000200E1600000200000001090000000200000001080000000500000000000000000000000000000000000000000000000000004000000000000000000000000000000040000000000000F03F00000000000000400000000000000840000000000000104000000000000008400102000000040000000000000000001040000000000000084000000000000010400000000000001440000000000000F03F000000000000104000000000000000000000000000000000010800000005000000333333333333FB3F000000000000F03F666666666666F63F9A9999999999D93F9A9999999999F93F9A9999999999D93F9A9999999999F93F000000000000E03F333333333333FB3F000000000000F03F",
        # "MULTICURVE((0 0,5 5),COMPOUNDCURVE((-1 -1,0 0),CIRCULARSTRING(0 0,1 1,2 0)),CIRCULARSTRING(4 0,4 4,8 4))",  # noqa: E501
        "010B000000030000000102000000020000000000000000000000000000000000000000000000000014400000000000001440010900000002000000010200000002000000000000000000F0BF000000000000F0BF0000000000000000000000000000000001080000000300000000000000000000000000000000000000000000000000F03F000000000000F03F00000000000000400000000000000000010800000003000000000000000000104000000000000000000000000000001040000000000000104000000000000020400000000000001040",
        # "MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0),(1 1,3 3,3 1,1 1)),((10 10,14 12,11 10,10 10),(11 11,11.5 11,11 11.5,11 11)))",  # noqa: E501
        "010C00000002000000010A000000020000000108000000050000000000000000000000000000000000000000000000000010400000000000000000000000000000104000000000000010400000000000000000000000000000104000000000000000000000000000000000010200000004000000000000000000F03F000000000000F03F000000000000084000000000000008400000000000000840000000000000F03F000000000000F03F000000000000F03F01030000000200000004000000000000000000244000000000000024400000000000002C40000000000000284000000000000026400000000000002440000000000000244000000000000024400400000000000000000026400000000000002640000000000000274000000000000026400000000000002640000000000000274000000000000026400000000000002640",
    ),
)
def test_from_wkb_nonlinear_unsupported(wkb):
    if shapely.geos_version >= (3, 13, 0):
        with pytest.raises(
            NotImplementedError,
            match="Nonlinear geometry types are not currently supported",
        ):
            shapely.from_wkb(wkb)

    else:
        # prior to GEOS 3.13 nonlinear types were rejected by GEOS on read from WKB
        with pytest.raises(shapely.errors.GEOSException, match="Unknown WKB type"):
            shapely.from_wkb(wkb)


def test_to_wkt():
    point = shapely.points(1, 1)
    actual = shapely.to_wkt(point)
    assert actual == "POINT (1 1)"

    actual = shapely.to_wkt(point, trim=False)
    assert actual == "POINT (1.000000 1.000000)"

    actual = shapely.to_wkt(point, rounding_precision=3, trim=False)
    assert actual == "POINT (1.000 1.000)"


def test_to_wkt_3D():
    # 3D points
    point_z = shapely.points(1, 1, 1)
    actual = shapely.to_wkt(point_z)
    assert actual == "POINT Z (1 1 1)"
    actual = shapely.to_wkt(point_z, output_dimension=3)
    assert actual == "POINT Z (1 1 1)"

    actual = shapely.to_wkt(point_z, output_dimension=2)
    assert actual == "POINT (1 1)"

    actual = shapely.to_wkt(point_z, old_3d=True)
    assert actual == "POINT (1 1 1)"


def test_to_wkt_none():
    # None propagates
    assert shapely.to_wkt(None) is None


def test_to_wkt_array_with_empty_z():
    # See GH-2004
    empty_wkt = ["POINT Z EMPTY", None, "POLYGON Z EMPTY"]
    empty_geoms = shapely.from_wkt(empty_wkt)
    if shapely.geos_version < (3, 9, 0):
        empty_wkt = ["POINT EMPTY", None, "POLYGON EMPTY"]
    assert list(shapely.to_wkt(empty_geoms)) == empty_wkt


def test_to_wkt_exceptions():
    with pytest.raises(TypeError):
        shapely.to_wkt(1)

    with pytest.raises(shapely.GEOSException):
        shapely.to_wkt(point, output_dimension=5)


def test_to_wkt_point_empty():
    assert shapely.to_wkt(empty_point) == "POINT EMPTY"


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="Empty geometries have no dimensionality on GEOS < 3.9",
)
@pytest.mark.parametrize(
    "wkt",
    [
        "POINT Z EMPTY",
        "LINESTRING Z EMPTY",
        "LINEARRING Z EMPTY",
        "POLYGON Z EMPTY",
    ],
)
def test_to_wkt_empty_z(wkt):
    assert shapely.to_wkt(shapely.from_wkt(wkt)) == wkt


def test_to_wkt_geometrycollection_with_point_empty():
    collection = shapely.geometrycollections([empty_point, point])
    # do not check the full value as some GEOS versions give
    # GEOMETRYCOLLECTION Z (...) and others give GEOMETRYCOLLECTION (...)
    assert shapely.to_wkt(collection).endswith("(POINT EMPTY, POINT (2 3))")


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="MULTIPOINT (EMPTY, (2 3)) only works for GEOS >= 3.9",
)
def test_to_wkt_multipoint_with_point_empty():
    geom = shapely.multipoints([empty_point, point])
    if shapely.geos_version >= (3, 12, 0):
        expected = "MULTIPOINT (EMPTY, (2 3))"
    else:
        # invalid WKT form
        expected = "MULTIPOINT (EMPTY, 2 3)"
    assert shapely.to_wkt(geom) == expected


@pytest.mark.skipif(
    shapely.geos_version >= (3, 9, 0),
    reason="MULTIPOINT (EMPTY, 2 3) gives ValueError on GEOS < 3.9",
)
def test_to_wkt_multipoint_with_point_empty_errors():
    # test if segfault is prevented
    geom = shapely.multipoints([empty_point, point])
    with pytest.raises(ValueError):
        shapely.to_wkt(geom)


def test_repr():
    assert repr(point) == "<POINT (2 3)>"


def test_repr_max_length():
    # the repr is limited to 80 characters
    geom = shapely.linestrings(np.arange(1000), np.arange(1000))
    representation = repr(geom)
    assert len(representation) == 80
    assert representation.endswith("...>")


@pytest.mark.skipif(
    shapely.geos_version >= (3, 9, 0),
    reason="MULTIPOINT (EMPTY, 2 3) gives Exception on GEOS < 3.9",
)
def test_repr_multipoint_with_point_empty():
    # Test if segfault is prevented
    geom = shapely.multipoints([point, empty_point])
    assert repr(geom) == "<shapely.MultiPoint Exception in WKT writer>"


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="Empty geometries have no dimensionality on GEOS < 3.9",
)
def test_repr_point_z_empty():
    assert repr(empty_point_z) == "<POINT Z EMPTY>"


def test_to_wkb():
    point = shapely.points(1, 1)
    actual = shapely.to_wkb(point, byte_order=1)
    assert actual == POINT11_WKB


def test_to_wkb_hex():
    point = shapely.points(1, 1)
    actual = shapely.to_wkb(point, hex=True, byte_order=1)
    le = "01"
    point_type = "01000000"
    coord = "000000000000F03F"  # 1.0 as double (LE)
    assert actual == le + point_type + 2 * coord


def test_to_wkb_3D():
    point_z = shapely.points(1, 1, 1)
    actual = shapely.to_wkb(point_z, byte_order=1)
    # fmt: off
    assert actual == b"\x01\x01\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"  # noqa
    # fmt: on
    actual = shapely.to_wkb(point_z, output_dimension=2, byte_order=1)
    assert actual == POINT11_WKB


def test_to_wkb_none():
    # None propagates
    assert shapely.to_wkb(None) is None


def test_to_wkb_exceptions():
    with pytest.raises(TypeError):
        shapely.to_wkb(1)

    with pytest.raises(shapely.GEOSException):
        shapely.to_wkb(point, output_dimension=5)

    with pytest.raises(ValueError):
        shapely.to_wkb(point, flavor="other")


def test_to_wkb_byte_order():
    point = shapely.points(1.0, 1.0)
    be = b"\x00"
    le = b"\x01"
    point_type = b"\x01\x00\x00\x00"  # 1 as 32-bit uint (LE)
    coord = b"\x00\x00\x00\x00\x00\x00\xf0?"  # 1.0 as double (LE)

    assert shapely.to_wkb(point, byte_order=1) == le + point_type + 2 * coord
    assert (
        shapely.to_wkb(point, byte_order=0) == be + point_type[::-1] + 2 * coord[::-1]
    )


def test_to_wkb_srid():
    # hex representation of POINT (0 0) with SRID=4
    ewkb = "01010000200400000000000000000000000000000000000000"
    wkb = "010100000000000000000000000000000000000000"

    actual = shapely.from_wkb(ewkb)
    assert shapely.to_wkt(actual, trim=True) == "POINT (0 0)"

    assert shapely.to_wkb(actual, hex=True, byte_order=1) == wkb
    assert shapely.to_wkb(actual, hex=True, include_srid=True, byte_order=1) == ewkb

    point = shapely.points(1, 1)
    point_with_srid = shapely.set_srid(point, np.int32(4326))
    result = shapely.to_wkb(point_with_srid, include_srid=True, byte_order=1)
    assert np.frombuffer(result[5:9], "<u4").item() == 4326


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10.0")
def test_to_wkb_flavor():
    # http://libgeos.org/specifications/wkb/#extended-wkb
    actual = shapely.to_wkb(point_z, byte_order=1)  # default "extended"
    assert actual.hex()[2:10] == "01000080"
    actual = shapely.to_wkb(point_z, byte_order=1, flavor="extended")
    assert actual.hex()[2:10] == "01000080"
    actual = shapely.to_wkb(point_z, byte_order=1, flavor="iso")
    assert actual.hex()[2:10] == "e9030000"


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10.0")
def test_to_wkb_flavor_srid():
    with pytest.raises(ValueError, match="cannot be used together"):
        shapely.to_wkb(point_z, include_srid=True, flavor="iso")


@pytest.mark.skipif(shapely.geos_version >= (3, 10, 0), reason="GEOS < 3.10.0")
def test_to_wkb_flavor_unsupported_geos():
    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.to_wkb(point_z, flavor="iso")


@pytest.mark.parametrize(
    "geom,expected",
    [
        (empty_point, POINT_NAN_WKB),
        (empty_point_z, POINT_NAN_WKB),
        (shapely.multipoints([empty_point]), MULTIPOINT_NAN_WKB),
        (shapely.multipoints([empty_point_z]), MULTIPOINT_NAN_WKB),
        (shapely.geometrycollections([empty_point]), GEOMETRYCOLLECTION_NAN_WKB),
        (shapely.geometrycollections([empty_point_z]), GEOMETRYCOLLECTION_NAN_WKB),
        (
            shapely.geometrycollections([shapely.multipoints([empty_point])]),
            NESTED_COLLECTION_NAN_WKB,
        ),
        (
            shapely.geometrycollections([shapely.multipoints([empty_point_z])]),
            NESTED_COLLECTION_NAN_WKB,
        ),
    ],
)
def test_to_wkb_point_empty_2d(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=2, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 16
    header_length = len(expected) - coordinate_length
    # Check the total length (this checks the correct dimensionality)
    assert len(actual) == header_length + coordinate_length
    # Check the header
    assert actual[:header_length] == expected[:header_length]
    # Check the coordinates (using numpy.isnan; there are many byte representations for NaN)
    assert np.isnan(struct.unpack("<2d", actual[header_length:])).all()


@pytest.mark.xfail(
    shapely.geos_version[:2] == (3, 8), reason="GEOS==3.8 never outputs 3D empty points"
)
@pytest.mark.parametrize(
    "geom,expected",
    [
        (empty_point_z, POINTZ_NAN_WKB),
        (shapely.multipoints([empty_point_z]), MULTIPOINTZ_NAN_WKB),
        (shapely.geometrycollections([empty_point_z]), GEOMETRYCOLLECTIONZ_NAN_WKB),
        (
            shapely.geometrycollections([shapely.multipoints([empty_point_z])]),
            NESTED_COLLECTIONZ_NAN_WKB,
        ),
    ],
)
def test_to_wkb_point_empty_3d(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=3, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 24
    header_length = len(expected) - coordinate_length
    # Check the total length (this checks the correct dimensionality)
    assert len(actual) == header_length + coordinate_length
    # Check the header
    assert actual[:header_length] == expected[:header_length]
    # Check the coordinates (using numpy.isnan; there are many byte representations for NaN)
    assert np.isnan(struct.unpack("<3d", actual[header_length:])).all()


@pytest.mark.xfail(
    shapely.geos_version < (3, 8, 0),
    reason="GEOS<3.8 always outputs 3D empty points if output_dimension=3",
)
@pytest.mark.parametrize(
    "geom,expected",
    [
        (empty_point, POINT_NAN_WKB),
        (shapely.multipoints([empty_point]), MULTIPOINT_NAN_WKB),
        (shapely.geometrycollections([empty_point]), GEOMETRYCOLLECTION_NAN_WKB),
        (
            shapely.geometrycollections([shapely.multipoints([empty_point])]),
            NESTED_COLLECTION_NAN_WKB,
        ),
    ],
)
def test_to_wkb_point_empty_2d_output_dim_3(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=3, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 16
    header_length = len(expected) - coordinate_length
    # Check the total length (this checks the correct dimensionality)
    assert len(actual) == header_length + coordinate_length
    # Check the header
    assert actual[:header_length] == expected[:header_length]
    # Check the coordinates (using numpy.isnan; there are many byte representations for NaN)
    assert np.isnan(struct.unpack("<2d", actual[header_length:])).all()


@pytest.mark.parametrize(
    "wkb,expected_type,expected_dim",
    [
        (POINT_NAN_WKB, 0, 2),
        (POINTZ_NAN_WKB, 0, 3),
        (MULTIPOINT_NAN_WKB, 4, 2),
        (MULTIPOINTZ_NAN_WKB, 4, 3),
        (GEOMETRYCOLLECTION_NAN_WKB, 7, 2),
        (GEOMETRYCOLLECTIONZ_NAN_WKB, 7, 3),
        (NESTED_COLLECTION_NAN_WKB, 7, 2),
        (NESTED_COLLECTIONZ_NAN_WKB, 7, 3),
    ],
)
def test_from_wkb_point_empty(wkb, expected_type, expected_dim):
    geom = shapely.from_wkb(wkb)
    # POINT (nan nan) transforms to an empty point
    assert shapely.is_empty(geom)
    assert shapely.get_type_id(geom) == expected_type
    # The dimensionality (2D/3D) is only read correctly for GEOS >= 3.9.0
    if shapely.geos_version >= (3, 9, 0):
        assert shapely.get_coordinate_dimension(geom) == expected_dim


def test_to_wkb_point_empty_srid():
    expected = shapely.set_srid(empty_point, 4236)
    wkb = shapely.to_wkb(expected, include_srid=True)
    actual = shapely.from_wkb(wkb)
    assert shapely.get_srid(actual) == 4236


@pytest.mark.parametrize("geom", all_types + (point_z, empty_point))
def test_pickle(geom):
    pickled = pickle.dumps(geom)
    assert_geometries_equal(pickle.loads(pickled), geom, tolerance=0)


@pytest.mark.parametrize("geom", all_types + (point_z, empty_point))
def test_pickle_with_srid(geom):
    geom = shapely.set_srid(geom, 4326)
    pickled = pickle.dumps(geom)
    assert shapely.get_srid(pickle.loads(pickled)) == 4326


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
@pytest.mark.parametrize(
    "geojson,expected",
    [
        (GEOJSON_GEOMETRY, GEOJSON_GEOMETRY_EXPECTED),
        (GEOJSON_FEATURE, GEOJSON_GEOMETRY_EXPECTED),
        (
            GEOJSON_FEATURECOLECTION,
            shapely.geometrycollections(GEOJSON_COLLECTION_EXPECTED),
        ),
        ([GEOJSON_GEOMETRY] * 2, [GEOJSON_GEOMETRY_EXPECTED] * 2),
        (None, None),
        ([GEOJSON_GEOMETRY, None], [GEOJSON_GEOMETRY_EXPECTED, None]),
    ],
)
def test_from_geojson(geojson, expected):
    actual = shapely.from_geojson(geojson)
    assert_geometries_equal(actual, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_exceptions():
    with pytest.raises(TypeError, match="Expected bytes or string, got int"):
        shapely.from_geojson(1)

    with pytest.raises(shapely.GEOSException, match="Error parsing JSON"):
        shapely.from_geojson("")

    with pytest.raises(shapely.GEOSException, match="Unknown geometry type"):
        shapely.from_geojson('{"type": "NoGeometry", "coordinates": []}')

    with pytest.raises(shapely.GEOSException, match="type must be array, but is null"):
        shapely.from_geojson('{"type": "LineString", "coordinates": null}')

    # Note: The two below tests are the reason that from_geojson is disabled for
    # GEOS 3.10.0 See https://trac.osgeo.org/geos/ticket/1138
    with pytest.raises(shapely.GEOSException, match="key 'type' not found"):
        shapely.from_geojson('{"geometry": null, "properties": []}')

    with pytest.raises(shapely.GEOSException, match="key 'type' not found"):
        shapely.from_geojson('{"no": "geojson"}')


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_warn_on_invalid():
    with pytest.warns(Warning, match="Invalid GeoJSON"):
        assert shapely.from_geojson("", on_invalid="warn") is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_ignore_on_invalid():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert shapely.from_geojson("", on_invalid="ignore") is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        shapely.from_geojson(GEOJSON_GEOMETRY, on_invalid="unsupported_option")


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "expected,geometry",
    [
        (GEOJSON_GEOMETRY, GEOJSON_GEOMETRY_EXPECTED),
        ([GEOJSON_GEOMETRY] * 2, [GEOJSON_GEOMETRY_EXPECTED] * 2),
        (None, None),
        ([GEOJSON_GEOMETRY, None], [GEOJSON_GEOMETRY_EXPECTED, None]),
    ],
)
def test_to_geojson(geometry, expected):
    actual = shapely.to_geojson(geometry, indent=4)
    assert np.all(actual == np.asarray(expected))


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("indent", [None, 0, 4])
def test_to_geojson_indent(indent):
    separators = (",", ":") if indent is None else (",", ": ")
    expected = json.dumps(
        json.loads(GEOJSON_GEOMETRY), indent=indent, separators=separators
    )
    actual = shapely.to_geojson(GEOJSON_GEOMETRY_EXPECTED, indent=indent)
    assert actual == expected


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_to_geojson_exceptions():
    with pytest.raises(TypeError):
        shapely.to_geojson(1)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geom",
    [
        empty_point,
        shapely.multipoints([empty_point, point]),
        shapely.geometrycollections([empty_point, point]),
        shapely.geometrycollections(
            [shapely.geometrycollections([empty_point]), point]
        ),
    ],
)
def test_to_geojson_point_empty(geom):
    # Pending GEOS ticket: https://trac.osgeo.org/geos/ticket/1139
    with pytest.raises(ValueError):
        assert shapely.to_geojson(geom)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
@pytest.mark.parametrize("geom", all_types)
def test_geojson_all_types(geom):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in GeoJSON")
    geojson = shapely.to_geojson(geom)
    actual = shapely.from_geojson(geojson)
    assert_geometries_equal(actual, geom)
