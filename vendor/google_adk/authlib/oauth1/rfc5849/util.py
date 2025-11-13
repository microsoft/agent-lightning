from authlib.common.urls import quote
from authlib.common.urls import unquote


def escape(s):
    return quote(s, safe=b"~")


def unescape(s):
    return unquote(s)
