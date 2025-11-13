"""authlib.oauth1.rfc5849.signature.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module represents a direct implementation of `section 3.4`_ of the spec.

.. _`section 3.4`: https://tools.ietf.org/html/rfc5849#section-3.4
"""

import binascii
import hashlib
import hmac

from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_unicode
from authlib.common.urls import urlparse

from .util import escape
from .util import unescape

SIGNATURE_HMAC_SHA1 = "HMAC-SHA1"
SIGNATURE_RSA_SHA1 = "RSA-SHA1"
SIGNATURE_PLAINTEXT = "PLAINTEXT"

SIGNATURE_TYPE_HEADER = "HEADER"
SIGNATURE_TYPE_QUERY = "QUERY"
SIGNATURE_TYPE_BODY = "BODY"


def construct_base_string(method, uri, params, host=None):
    """Generate signature base string from request, per `Section 3.4.1`_.

    For example, the HTTP request::

        POST /request?b5=%3D%253D&a3=a&c%40=&a2=r%20b HTTP/1.1
        Host: example.com
        Content-Type: application/x-www-form-urlencoded
        Authorization: OAuth realm="Example",
            oauth_consumer_key="9djdj82h48djs9d2",
            oauth_token="kkk9d7dh3k39sjv7",
            oauth_signature_method="HMAC-SHA1",
            oauth_timestamp="137131201",
            oauth_nonce="7d8f3e4a",
            oauth_signature="bYT5CMsGcbgUdFHObYMEfcx6bsw%3D"

        c2&a3=2+q

    is represented by the following signature base string (line breaks
    are for display purposes only)::

        POST&http%3A%2F%2Fexample.com%2Frequest&a2%3Dr%2520b%26a3%3D2%2520q
        %26a3%3Da%26b5%3D%253D%25253D%26c%2540%3D%26c2%3D%26oauth_consumer_
        key%3D9djdj82h48djs9d2%26oauth_nonce%3D7d8f3e4a%26oauth_signature_m
        ethod%3DHMAC-SHA1%26oauth_timestamp%3D137131201%26oauth_token%3Dkkk
        9d7dh3k39sjv7

    .. _`Section 3.4.1`: https://tools.ietf.org/html/rfc5849#section-3.4.1
    """
    # Create base string URI per Section 3.4.1.2
    base_string_uri = normalize_base_string_uri(uri, host)

    # Cleanup parameter sources per Section 3.4.1.3.1
    unescaped_params = []
    for k, v in params:
        # The "oauth_signature" parameter MUST be excluded from the signature
        if k in ("oauth_signature", "realm"):
            continue

        # ensure oauth params are unescaped
        if k.startswith("oauth_"):
            v = unescape(v)
        unescaped_params.append((k, v))

    # Normalize parameters per Section 3.4.1.3.2
    normalized_params = normalize_parameters(unescaped_params)

    # construct base string
    return "&".join(
        [
            escape(method.upper()),
            escape(base_string_uri),
            escape(normalized_params),
        ]
    )


def normalize_base_string_uri(uri, host=None):
    """Normalize Base String URI per `Section 3.4.1.2`_.

    For example, the HTTP request::

        GET /r%20v/X?id=123 HTTP/1.1
        Host: EXAMPLE.COM:80

    is represented by the base string URI: "http://example.com/r%20v/X".

    In another example, the HTTPS request::

        GET /?q=1 HTTP/1.1
        Host: www.example.net:8080

    is represented by the base string URI: "https://www.example.net:8080/".

    .. _`Section 3.4.1.2`: https://tools.ietf.org/html/rfc5849#section-3.4.1.2

    The host argument overrides the netloc part of the uri argument.
    """
    uri = to_unicode(uri)
    scheme, netloc, path, params, query, fragment = urlparse.urlparse(uri)

    # The scheme, authority, and path of the request resource URI `RFC3986`
    # are included by constructing an "http" or "https" URI representing
    # the request resource (without the query or fragment) as follows:
    #
    # .. _`RFC3986`: https://tools.ietf.org/html/rfc3986

    if not scheme or not netloc:
        raise ValueError("uri must include a scheme and netloc")

    # Per `RFC 2616 section 5.1.2`_:
    #
    # Note that the absolute path cannot be empty; if none is present in
    # the original URI, it MUST be given as "/" (the server root).
    #
    # .. _`RFC 2616 section 5.1.2`: https://tools.ietf.org/html/rfc2616#section-5.1.2
    if not path:
        path = "/"

    # 1.  The scheme and host MUST be in lowercase.
    scheme = scheme.lower()
    netloc = netloc.lower()

    # 2.  The host and port values MUST match the content of the HTTP
    #     request "Host" header field.
    if host is not None:
        netloc = host.lower()

    # 3.  The port MUST be included if it is not the default port for the
    #     scheme, and MUST be excluded if it is the default.  Specifically,
    #     the port MUST be excluded when making an HTTP request `RFC2616`_
    #     to port 80 or when making an HTTPS request `RFC2818`_ to port 443.
    #     All other non-default port numbers MUST be included.
    #
    # .. _`RFC2616`: https://tools.ietf.org/html/rfc2616
    # .. _`RFC2818`: https://tools.ietf.org/html/rfc2818
    default_ports = (
        ("http", "80"),
        ("https", "443"),
    )
    if ":" in netloc:
        host, port = netloc.split(":", 1)
        if (scheme, port) in default_ports:
            netloc = host

    return urlparse.urlunparse((scheme, netloc, path, params, "", ""))


def normalize_parameters(params):
    """Normalize parameters per `Section 3.4.1.3.2`_.

    For example, the list of parameters from the previous section would
    be normalized as follows:

    Encoded::

    +------------------------+------------------+
    |          Name          |       Value      |
    +------------------------+------------------+
    |           b5           |     %3D%253D     |
    |           a3           |         a        |
    |          c%40          |                  |
    |           a2           |       r%20b      |
    |   oauth_consumer_key   | 9djdj82h48djs9d2 |
    |       oauth_token      | kkk9d7dh3k39sjv7 |
    | oauth_signature_method |     HMAC-SHA1    |
    |     oauth_timestamp    |     137131201    |
    |       oauth_nonce      |     7d8f3e4a     |
    |           c2           |                  |
    |           a3           |       2%20q      |
    +------------------------+------------------+

    Sorted::

    +------------------------+------------------+
    |          Name          |       Value      |
    +------------------------+------------------+
    |           a2           |       r%20b      |
    |           a3           |       2%20q      |
    |           a3           |         a        |
    |           b5           |     %3D%253D     |
    |          c%40          |                  |
    |           c2           |                  |
    |   oauth_consumer_key   | 9djdj82h48djs9d2 |
    |       oauth_nonce      |     7d8f3e4a     |
    | oauth_signature_method |     HMAC-SHA1    |
    |     oauth_timestamp    |     137131201    |
    |       oauth_token      | kkk9d7dh3k39sjv7 |
    +------------------------+------------------+

    Concatenated Pairs::

    +-------------------------------------+
    |              Name=Value             |
    +-------------------------------------+
    |               a2=r%20b              |
    |               a3=2%20q              |
    |                 a3=a                |
    |             b5=%3D%253D             |
    |                c%40=                |
    |                 c2=                 |
    | oauth_consumer_key=9djdj82h48djs9d2 |
    |         oauth_nonce=7d8f3e4a        |
    |   oauth_signature_method=HMAC-SHA1  |
    |      oauth_timestamp=137131201      |
    |     oauth_token=kkk9d7dh3k39sjv7    |
    +-------------------------------------+

    and concatenated together into a single string (line breaks are for
    display purposes only)::

        a2=r%20b&a3=2%20q&a3=a&b5=%3D%253D&c%40=&c2=&oauth_consumer_key=9dj
        dj82h48djs9d2&oauth_nonce=7d8f3e4a&oauth_signature_method=HMAC-SHA1
        &oauth_timestamp=137131201&oauth_token=kkk9d7dh3k39sjv7

    .. _`Section 3.4.1.3.2`: https://tools.ietf.org/html/rfc5849#section-3.4.1.3.2
    """
    # 1.  First, the name and value of each parameter are encoded
    #     (`Section 3.6`_).
    #
    # .. _`Section 3.6`: https://tools.ietf.org/html/rfc5849#section-3.6
    key_values = [(escape(k), escape(v)) for k, v in params]

    # 2.  The parameters are sorted by name, using ascending byte value
    #     ordering.  If two or more parameters share the same name, they
    #     are sorted by their value.
    key_values.sort()

    # 3.  The name of each parameter is concatenated to its corresponding
    #     value using an "=" character (ASCII code 61) as a separator, even
    #     if the value is empty.
    parameter_parts = [f"{k}={v}" for k, v in key_values]

    # 4.  The sorted name/value pairs are concatenated together into a
    #     single string by using an "&" character (ASCII code 38) as
    #     separator.
    return "&".join(parameter_parts)


def generate_signature_base_string(request):
    """Generate signature base string from request."""
    host = request.headers.get("Host", None)
    return construct_base_string(request.method, request.uri, request.params, host)


def hmac_sha1_signature(base_string, client_secret, token_secret):
    """Generate signature via HMAC-SHA1 method, per `Section 3.4.2`_.

    The "HMAC-SHA1" signature method uses the HMAC-SHA1 signature
    algorithm as defined in `RFC2104`_::

        digest = HMAC - SHA1(key, text)

    .. _`RFC2104`: https://tools.ietf.org/html/rfc2104
    .. _`Section 3.4.2`: https://tools.ietf.org/html/rfc5849#section-3.4.2
    """
    # The HMAC-SHA1 function variables are used in following way:

    # text is set to the value of the signature base string from
    # `Section 3.4.1.1`_.
    #
    # .. _`Section 3.4.1.1`: https://tools.ietf.org/html/rfc5849#section-3.4.1.1
    text = base_string

    # key is set to the concatenated values of:
    # 1.  The client shared-secret, after being encoded (`Section 3.6`_).
    #
    # .. _`Section 3.6`: https://tools.ietf.org/html/rfc5849#section-3.6
    key = escape(client_secret or "")

    # 2.  An "&" character (ASCII code 38), which MUST be included
    #     even when either secret is empty.
    key += "&"

    # 3.  The token shared-secret, after being encoded (`Section 3.6`_).
    #
    # .. _`Section 3.6`: https://tools.ietf.org/html/rfc5849#section-3.6
    key += escape(token_secret or "")

    signature = hmac.new(to_bytes(key), to_bytes(text), hashlib.sha1)

    # digest  is used to set the value of the "oauth_signature" protocol
    #         parameter, after the result octet string is base64-encoded
    #         per `RFC2045, Section 6.8`.
    #
    # .. _`RFC2045, Section 6.8`: https://tools.ietf.org/html/rfc2045#section-6.8
    sig = binascii.b2a_base64(signature.digest())[:-1]
    return to_unicode(sig)


def rsa_sha1_signature(base_string, rsa_private_key):
    """Generate signature via RSA-SHA1 method, per `Section 3.4.3`_.

    The "RSA-SHA1" signature method uses the RSASSA-PKCS1-v1_5 signature
    algorithm as defined in `RFC3447, Section 8.2`_ (also known as
    PKCS#1), using SHA-1 as the hash function for EMSA-PKCS1-v1_5.  To
    use this method, the client MUST have established client credentials
    with the server that included its RSA public key (in a manner that is
    beyond the scope of this specification).

    .. _`Section 3.4.3`: https://tools.ietf.org/html/rfc5849#section-3.4.3
    .. _`RFC3447, Section 8.2`: https://tools.ietf.org/html/rfc3447#section-8.2
    """
    from .rsa import sign_sha1

    base_string = to_bytes(base_string)
    s = sign_sha1(to_bytes(base_string), rsa_private_key)
    sig = binascii.b2a_base64(s)[:-1]
    return to_unicode(sig)


def plaintext_signature(client_secret, token_secret):
    """Generate signature via PLAINTEXT method, per `Section 3.4.4`_.

    The "PLAINTEXT" method does not employ a signature algorithm.  It
    MUST be used with a transport-layer mechanism such as TLS or SSL (or
    sent over a secure channel with equivalent protections).  It does not
    utilize the signature base string or the "oauth_timestamp" and
    "oauth_nonce" parameters.

    .. _`Section 3.4.4`: https://tools.ietf.org/html/rfc5849#section-3.4.4
    """
    # The "oauth_signature" protocol parameter is set to the concatenated
    # value of:

    # 1.  The client shared-secret, after being encoded (`Section 3.6`_).
    #
    # .. _`Section 3.6`: https://tools.ietf.org/html/rfc5849#section-3.6
    signature = escape(client_secret or "")

    # 2.  An "&" character (ASCII code 38), which MUST be included even
    #     when either secret is empty.
    signature += "&"

    # 3.  The token shared-secret, after being encoded (`Section 3.6`_).
    #
    # .. _`Section 3.6`: https://tools.ietf.org/html/rfc5849#section-3.6
    signature += escape(token_secret or "")

    return signature


def sign_hmac_sha1(client, request):
    """Sign a HMAC-SHA1 signature."""
    base_string = generate_signature_base_string(request)
    return hmac_sha1_signature(base_string, client.client_secret, client.token_secret)


def sign_rsa_sha1(client, request):
    """Sign a RSASSA-PKCS #1 v1.5 base64 encoded signature."""
    base_string = generate_signature_base_string(request)
    return rsa_sha1_signature(base_string, client.rsa_key)


def sign_plaintext(client, request):
    """Sign a PLAINTEXT signature."""
    return plaintext_signature(client.client_secret, client.token_secret)


def verify_hmac_sha1(request):
    """Verify a HMAC-SHA1 signature."""
    base_string = generate_signature_base_string(request)
    sig = hmac_sha1_signature(base_string, request.client_secret, request.token_secret)
    return hmac.compare_digest(sig, request.signature)


def verify_rsa_sha1(request):
    """Verify a RSASSA-PKCS #1 v1.5 base64 encoded signature."""
    from .rsa import verify_sha1

    base_string = generate_signature_base_string(request)
    sig = binascii.a2b_base64(to_bytes(request.signature))
    return verify_sha1(sig, to_bytes(base_string), request.rsa_public_key)


def verify_plaintext(request):
    """Verify a PLAINTEXT signature."""
    sig = plaintext_signature(request.client_secret, request.token_secret)
    return hmac.compare_digest(sig, request.signature)
