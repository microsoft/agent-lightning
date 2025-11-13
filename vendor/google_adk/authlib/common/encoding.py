import base64
import json
import struct


def to_bytes(x, charset="utf-8", errors="strict"):
    if x is None:
        return None
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode(charset, errors)
    if isinstance(x, (int, float)):
        return str(x).encode(charset, errors)
    return bytes(x)


def to_unicode(x, charset="utf-8", errors="strict"):
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode(charset, errors)
    return str(x)


def to_native(x, encoding="ascii"):
    if isinstance(x, str):
        return x
    return x.decode(encoding)


def json_loads(s):
    return json.loads(s)


def json_dumps(data, ensure_ascii=False):
    return json.dumps(data, ensure_ascii=ensure_ascii, separators=(",", ":"))


def urlsafe_b64decode(s):
    s += b"=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def urlsafe_b64encode(s):
    return base64.urlsafe_b64encode(s).rstrip(b"=")


def base64_to_int(s):
    data = urlsafe_b64decode(to_bytes(s, charset="ascii"))
    buf = struct.unpack(f"{len(data)}B", data)
    return int("".join([f"{byte:02x}" for byte in buf]), 16)


def int_to_base64(num):
    if num < 0:
        raise ValueError("Must be a positive integer")

    s = num.to_bytes((num.bit_length() + 7) // 8, "big", signed=False)
    return to_unicode(urlsafe_b64encode(s))


def json_b64encode(text):
    if isinstance(text, dict):
        text = json_dumps(text)
    return urlsafe_b64encode(to_bytes(text))
