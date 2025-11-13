"""authlib.spec.rfc5849.parameters.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains methods related to `section 3.5`_ of the OAuth 1.0a spec.

.. _`section 3.5`: https://tools.ietf.org/html/rfc5849#section-3.5
"""

from authlib.common.urls import extract_params
from authlib.common.urls import url_encode
from authlib.common.urls import urlparse

from .util import escape


def prepare_headers(oauth_params, headers=None, realm=None):
    """**Prepare the Authorization header.**
    Per `section 3.5.1`_ of the spec.

    Protocol parameters can be transmitted using the HTTP "Authorization"
    header field as defined by `RFC2617`_ with the auth-scheme name set to
    "OAuth" (case insensitive).

    For example::

        Authorization: OAuth realm="Photos",
            oauth_consumer_key="dpf43f3p2l4k3l03",
            oauth_signature_method="HMAC-SHA1",
            oauth_timestamp="137131200",
            oauth_nonce="wIjqoS",
            oauth_callback="http%3A%2F%2Fprinter.example.com%2Fready",
            oauth_signature="74KNZJeDHnMBp0EMJ9ZHt%2FXKycU%3D",
            oauth_version="1.0"

    .. _`section 3.5.1`: https://tools.ietf.org/html/rfc5849#section-3.5.1
    .. _`RFC2617`: https://tools.ietf.org/html/rfc2617
    """
    headers = headers or {}

    # step 1, 2, 3 in Section 3.5.1
    header_parameters = ", ".join(
        [
            f'{escape(k)}="{escape(v)}"'
            for k, v in oauth_params
            if k.startswith("oauth_")
        ]
    )

    # 4.  The OPTIONAL "realm" parameter MAY be added and interpreted per
    #     `RFC2617 section 1.2`_.
    #
    # .. _`RFC2617 section 1.2`: https://tools.ietf.org/html/rfc2617#section-1.2
    if realm:
        # NOTE: realm should *not* be escaped
        header_parameters = f'realm="{realm}", ' + header_parameters

    # the auth-scheme name set to "OAuth" (case insensitive).
    headers["Authorization"] = f"OAuth {header_parameters}"
    return headers


def _append_params(oauth_params, params):
    """Append OAuth params to an existing set of parameters.

    Both params and oauth_params is must be lists of 2-tuples.

    Per `section 3.5.2`_ and `3.5.3`_ of the spec.

    .. _`section 3.5.2`: https://tools.ietf.org/html/rfc5849#section-3.5.2
    .. _`3.5.3`: https://tools.ietf.org/html/rfc5849#section-3.5.3

    """
    merged = list(params)
    merged.extend(oauth_params)
    # The request URI / entity-body MAY include other request-specific
    # parameters, in which case, the protocol parameters SHOULD be appended
    # following the request-specific parameters, properly separated by an "&"
    # character (ASCII code 38)
    merged.sort(key=lambda i: i[0].startswith("oauth_"))
    return merged


def prepare_form_encoded_body(oauth_params, body):
    """Prepare the Form-Encoded Body.

    Per `section 3.5.2`_ of the spec.

    .. _`section 3.5.2`: https://tools.ietf.org/html/rfc5849#section-3.5.2

    """
    # append OAuth params to the existing body
    return url_encode(_append_params(oauth_params, body))


def prepare_request_uri_query(oauth_params, uri):
    """Prepare the Request URI Query.

    Per `section 3.5.3`_ of the spec.

    .. _`section 3.5.3`: https://tools.ietf.org/html/rfc5849#section-3.5.3

    """
    # append OAuth params to the existing set of query components
    sch, net, path, par, query, fra = urlparse.urlparse(uri)
    query = url_encode(_append_params(oauth_params, extract_params(query) or []))
    return urlparse.urlunparse((sch, net, path, par, query, fra))
