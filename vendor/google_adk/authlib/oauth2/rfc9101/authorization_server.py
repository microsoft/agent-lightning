from authlib.jose import jwt
from authlib.jose.errors import JoseError

from ..rfc6749 import AuthorizationServer
from ..rfc6749 import ClientMixin
from ..rfc6749 import InvalidRequestError
from ..rfc6749.authenticate_client import _validate_client
from ..rfc6749.requests import BasicOAuth2Payload
from ..rfc6749.requests import OAuth2Request
from .errors import InvalidRequestObjectError
from .errors import InvalidRequestUriError
from .errors import RequestNotSupportedError
from .errors import RequestUriNotSupportedError


class JWTAuthenticationRequest:
    """Authorization server extension implementing the support
    for JWT secured authentication request, as defined in :rfc:`RFC9101 <9101>`.

    :param support_request: Whether to enable support for the ``request`` parameter.
    :param support_request_uri: Whether to enable support for the ``request_uri`` parameter.

    This extension is intended to be inherited and registered into the authorization server::

        class JWTAuthenticationRequest(rfc9101.JWTAuthenticationRequest):
            def resolve_client_public_key(self, client: ClientMixin):
                return get_jwks_for_client(client)

            def get_request_object(self, request_uri: str):
                try:
                    return requests.get(request_uri).text
                except requests.Exception:
                    return None

            def get_server_metadata(self):
                return {
                    "issuer": ...,
                    "authorization_endpoint": ...,
                    "require_signed_request_object": ...,
                }

            def get_client_require_signed_request_object(self, client: ClientMixin):
                return client.require_signed_request_object


        authorization_server.register_extension(JWTAuthenticationRequest())
    """

    def __init__(self, support_request: bool = True, support_request_uri: bool = True):
        self.support_request = support_request
        self.support_request_uri = support_request_uri

    def __call__(self, authorization_server: AuthorizationServer):
        authorization_server.register_hook(
            "before_get_authorization_grant", self.parse_authorization_request
        )

    def parse_authorization_request(
        self, authorization_server: AuthorizationServer, request: OAuth2Request
    ):
        client = _validate_client(
            authorization_server.query_client, request.payload.client_id
        )
        if not self._shoud_proceed_with_request_object(
            authorization_server, request, client
        ):
            return

        raw_request_object = self._get_raw_request_object(authorization_server, request)
        request_object = self._decode_request_object(
            request, client, raw_request_object
        )
        payload = BasicOAuth2Payload(request_object)
        request.payload = payload

    def _shoud_proceed_with_request_object(
        self,
        authorization_server: AuthorizationServer,
        request: OAuth2Request,
        client: ClientMixin,
    ) -> bool:
        if "request" in request.payload.data and "request_uri" in request.payload.data:
            raise InvalidRequestError(
                "The 'request' and 'request_uri' parameters are mutually exclusive.",
                state=request.payload.state,
            )

        if "request" in request.payload.data:
            if not self.support_request:
                raise RequestNotSupportedError(state=request.payload.state)
            return True

        if "request_uri" in request.payload.data:
            if not self.support_request_uri:
                raise RequestUriNotSupportedError(state=request.payload.state)
            return True

        # When the value of it [require_signed_request_object] as client metadata is true,
        # then the server MUST reject the authorization request
        # from the client that does not conform to this specification.
        if self.get_client_require_signed_request_object(client):
            raise InvalidRequestError(
                "Authorization requests for this client must use signed request objects.",
                state=request.payload.state,
            )

        # When the value of it [require_signed_request_object] as server metadata is true,
        # then the server MUST reject the authorization request
        # from any client that does not conform to this specification.
        metadata = self.get_server_metadata()
        if metadata and metadata.get("require_signed_request_object", False):
            raise InvalidRequestError(
                "Authorization requests for this server must use signed request objects.",
                state=request.payload.state,
            )

        return False

    def _get_raw_request_object(
        self, authorization_server: AuthorizationServer, request: OAuth2Request
    ) -> str:
        if "request_uri" in request.payload.data:
            raw_request_object = self.get_request_object(
                request.payload.data["request_uri"]
            )
            if not raw_request_object:
                raise InvalidRequestUriError(state=request.payload.state)

        else:
            raw_request_object = request.payload.data["request"]

        return raw_request_object

    def _decode_request_object(
        self, request, client: ClientMixin, raw_request_object: str
    ):
        jwks = self.resolve_client_public_key(client)

        try:
            request_object = jwt.decode(raw_request_object, jwks)
            request_object.validate()

        except JoseError as error:
            raise InvalidRequestObjectError(
                description=error.description or InvalidRequestObjectError.description,
                state=request.payload.state,
            ) from error

        # It MUST also reject the request if the Request Object uses an
        # alg value of none when this server metadata value is true.
        # If omitted, the default value is false.
        if (
            self.get_client_require_signed_request_object(client)
            and request_object.header["alg"] == "none"
        ):
            raise InvalidRequestError(
                "Authorization requests for this client must use signed request objects.",
                state=request.payload.state,
            )

        # It MUST also reject the request if the Request Object uses an
        # alg value of none. If omitted, the default value is false.
        metadata = self.get_server_metadata()
        if (
            metadata
            and metadata.get("require_signed_request_object", False)
            and request_object.header["alg"] == "none"
        ):
            raise InvalidRequestError(
                "Authorization requests for this server must use signed request objects.",
                state=request.payload.state,
            )

        # The client ID values in the client_id request parameter and in
        # the Request Object client_id claim MUST be identical.
        if request_object["client_id"] != request.payload.client_id:
            raise InvalidRequestError(
                "The 'client_id' claim from the request parameters "
                "and the request object claims don't match.",
                state=request.payload.state,
            )

        # The Request Object MAY be sent by value, as described in Section 5.1,
        # or by reference, as described in Section 5.2. request and
        # request_uri parameters MUST NOT be included in Request Objects.
        if "request" in request_object or "request_uri" in request_object:
            raise InvalidRequestError(
                "The 'request' and 'request_uri' parameters must not be included in the request object.",
                state=request.payload.state,
            )

        return request_object

    def get_request_object(self, request_uri: str):
        """Download the request object at ``request_uri``.

        This method must be implemented if the ``request_uri`` parameter is supported::

            class JWTAuthenticationRequest(rfc9101.JWTAuthenticationRequest):
                def get_request_object(self, request_uri: str):
                    try:
                        return requests.get(request_uri).text
                    except requests.Exception:
                        return None
        """
        raise NotImplementedError()

    def resolve_client_public_keys(self, client: ClientMixin):
        """Resolve the client public key for verifying the JWT signature.
        A client may have many public keys, in this case, we can retrieve it
        via ``kid`` value in headers. Developers MUST implement this method::

            class JWTAuthenticationRequest(rfc9101.JWTAuthenticationRequest):
                def resolve_client_public_key(self, client):
                    if client.jwks_uri:
                        return requests.get(client.jwks_uri).json

                    return client.jwks
        """
        raise NotImplementedError()

    def get_server_metadata(self) -> dict:
        """Return server metadata which includes supported grant types,
        response types and etc.

        When the ``require_signed_request_object`` claim is :data:`True`,
        all clients require that authorization requests
        use request objects, and an error will be returned when the authorization
        request payload is passed in the request body or query string::

            class JWTAuthenticationRequest(rfc9101.JWTAuthenticationRequest):
                def get_server_metadata(self):
                    return {
                        "issuer": ...,
                        "authorization_endpoint": ...,
                        "require_signed_request_object": ...,
                    }

        """
        return {}  # pragma: no cover

    def get_client_require_signed_request_object(self, client: ClientMixin) -> bool:
        """Return the 'require_signed_request_object' client metadata.

        When :data:`True`, the client requires that authorization requests
        use request objects, and an error will be returned when the authorization
        request payload is passed in the request body or query string::

           class JWTAuthenticationRequest(rfc9101.JWTAuthenticationRequest):
               def get_client_require_signed_request_object(self, client):
                   return client.require_signed_request_object

        If not implemented, the value is considered as :data:`False`.
        """
        return False  # pragma: no cover
