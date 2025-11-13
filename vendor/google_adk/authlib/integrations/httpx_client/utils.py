from httpx import Request

HTTPX_CLIENT_KWARGS = [
    "headers",
    "cookies",
    "verify",
    "cert",
    "http1",
    "http2",
    "proxy",
    "mounts",
    "timeout",
    "follow_redirects",
    "limits",
    "max_redirects",
    "event_hooks",
    "base_url",
    "transport",
    "trust_env",
    "default_encoding",
]


def extract_client_kwargs(kwargs):
    client_kwargs = {}
    for k in HTTPX_CLIENT_KWARGS:
        if k in kwargs:
            client_kwargs[k] = kwargs.pop(k)
    return client_kwargs


def build_request(url, headers, body, initial_request: Request) -> Request:
    """Make sure that all the data from initial request is passed to the updated object."""
    updated_request = Request(
        method=initial_request.method, url=url, headers=headers, content=body
    )

    if hasattr(initial_request, "extensions"):
        updated_request.extensions = initial_request.extensions

    return updated_request
