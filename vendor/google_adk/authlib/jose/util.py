import binascii

from authlib.common.encoding import json_loads
from authlib.common.encoding import to_unicode
from authlib.common.encoding import urlsafe_b64decode
from authlib.jose.errors import DecodeError


def extract_header(header_segment, error_cls):
    if len(header_segment) > 256000:
        raise ValueError("Value of header is too long")

    header_data = extract_segment(header_segment, error_cls, "header")

    try:
        header = json_loads(header_data.decode("utf-8"))
    except ValueError as e:
        raise error_cls(f"Invalid header string: {e}") from e

    if not isinstance(header, dict):
        raise error_cls("Header must be a json object")
    return header


def extract_segment(segment, error_cls, name="payload"):
    if len(segment) > 256000:
        raise ValueError(f"Value of {name} is too long")

    try:
        return urlsafe_b64decode(segment)
    except (TypeError, binascii.Error) as exc:
        msg = f"Invalid {name} padding"
        raise error_cls(msg) from exc


def ensure_dict(s, structure_name):
    if not isinstance(s, dict):
        try:
            s = json_loads(to_unicode(s))
        except (ValueError, TypeError) as exc:
            raise DecodeError(f"Invalid {structure_name}") from exc

    if not isinstance(s, dict):
        raise DecodeError(f"Invalid {structure_name}")

    return s
