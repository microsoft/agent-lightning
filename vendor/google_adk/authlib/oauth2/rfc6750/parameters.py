from authlib.common.urls import add_params_to_qs
from authlib.common.urls import add_params_to_uri


def add_to_uri(token, uri):
    """Add a Bearer Token to the request URI.
    Not recommended, use only if client can't use authorization header or body.

    http://www.example.com/path?access_token=h480djs93hd8
    """
    return add_params_to_uri(uri, [("access_token", token)])


def add_to_headers(token, headers=None):
    """Add a Bearer Token to the request URI.
    Recommended method of passing bearer tokens.

    Authorization: Bearer h480djs93hd8
    """
    headers = headers or {}
    headers["Authorization"] = f"Bearer {token}"
    return headers


def add_to_body(token, body=None):
    """Add a Bearer Token to the request body.

    access_token=h480djs93hd8
    """
    if body is None:
        body = ""
    return add_params_to_qs(body, [("access_token", token)])


def add_bearer_token(token, uri, headers, body, placement="header"):
    if placement in ("uri", "url", "query"):
        uri = add_to_uri(token, uri)
    elif placement in ("header", "headers"):
        headers = add_to_headers(token, headers)
    elif placement == "body":
        body = add_to_body(token, body)
    return uri, headers, body
