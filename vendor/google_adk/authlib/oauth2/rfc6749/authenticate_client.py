"""authlib.oauth2.rfc6749.authenticate_client.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registry of client authentication methods, with 3 built-in methods:

1. client_secret_basic
2. client_secret_post
3. none

The "client_secret_basic" method is used a lot in examples of `RFC6749`_,
but the concept of naming are introduced in `RFC7591`_.

.. _`RFC6749`: https://tools.ietf.org/html/rfc6749
.. _`RFC7591`: https://tools.ietf.org/html/rfc7591
"""

import logging

from .errors import InvalidClientError
from .util import extract_basic_authorization

log = logging.getLogger(__name__)

__all__ = ["ClientAuthentication"]


class ClientAuthentication:
    def __init__(self, query_client):
        self.query_client = query_client
        self._methods = {
            "none": authenticate_none,
            "client_secret_basic": authenticate_client_secret_basic,
            "client_secret_post": authenticate_client_secret_post,
        }

    def register(self, method, func):
        self._methods[method] = func

    def authenticate(self, request, methods, endpoint):
        for method in methods:
            func = self._methods[method]
            client = func(self.query_client, request)
            if client and client.check_endpoint_auth_method(method, endpoint):
                request.auth_method = method
                return client

        if "client_secret_basic" in methods:
            raise InvalidClientError(
                status_code=401,
                description=f"The client cannot authenticate with methods: {methods}",
            )
        raise InvalidClientError(
            description=f"The client cannot authenticate with methods: {methods}",
        )

    def __call__(self, request, methods, endpoint="token"):
        return self.authenticate(request, methods, endpoint)


def authenticate_client_secret_basic(query_client, request):
    """Authenticate client by ``client_secret_basic`` method. The client
    uses HTTP Basic for authentication.
    """
    client_id, client_secret = extract_basic_authorization(request.headers)
    if client_id and client_secret:
        client = _validate_client(query_client, client_id, 401)
        if client.check_client_secret(client_secret):
            log.debug(f'Authenticate {client_id} via "client_secret_basic" success')
            return client
    log.debug(f'Authenticate {client_id} via "client_secret_basic" failed')


def authenticate_client_secret_post(query_client, request):
    """Authenticate client by ``client_secret_post`` method. The client
    uses POST parameters for authentication.
    """
    data = request.form
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    if client_id and client_secret:
        client = _validate_client(query_client, client_id)
        if client.check_client_secret(client_secret):
            log.debug(f'Authenticate {client_id} via "client_secret_post" success')
            return client
    log.debug(f'Authenticate {client_id} via "client_secret_post" failed')


def authenticate_none(query_client, request):
    """Authenticate public client by ``none`` method. The client
    does not have a client secret.
    """
    client_id = request.payload.client_id
    if client_id and not request.payload.data.get("client_secret"):
        client = _validate_client(query_client, client_id)
        log.debug(f'Authenticate {client_id} via "none" success')
        return client
    log.debug(f'Authenticate {client_id} via "none" failed')


def _validate_client(query_client, client_id, status_code=400):
    if client_id is None:
        raise InvalidClientError(
            status_code=status_code,
            description="Missing 'client_id' parameter.",
        )

    client = query_client(client_id)
    if not client:
        raise InvalidClientError(
            status_code=status_code,
            description="The client does not exist on this server.",
        )

    return client
