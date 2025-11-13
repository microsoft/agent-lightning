import binascii
import os
import time

from authlib.common.security import generate_token
from authlib.consts import default_json_headers
from authlib.deprecate import deprecate
from authlib.jose import JoseError
from authlib.jose import JsonWebToken

from ..rfc6749 import AccessDeniedError
from ..rfc6749 import InvalidRequestError
from .claims import ClientMetadataClaims
from .errors import InvalidClientMetadataError
from .errors import InvalidSoftwareStatementError
from .errors import UnapprovedSoftwareStatementError


class ClientRegistrationEndpoint:
    """The client registration endpoint is an OAuth 2.0 endpoint designed to
    allow a client to be registered with the authorization server.
    """

    ENDPOINT_NAME = "client_registration"

    #: Rewrite this value with a list to support ``software_statement``
    #: e.g. ``software_statement_alg_values_supported = ['RS256']``
    software_statement_alg_values_supported = None

    def __init__(self, server=None, claims_classes=None):
        self.server = server
        self.claims_classes = claims_classes or [ClientMetadataClaims]

    def __call__(self, request):
        return self.create_registration_response(request)

    def create_registration_response(self, request):
        token = self.authenticate_token(request)
        if not token:
            raise AccessDeniedError()

        request.credential = token

        client_metadata = self.extract_client_metadata(request)
        client_info = self.generate_client_info(request)
        body = {}
        body.update(client_metadata)
        body.update(client_info)
        client = self.save_client(client_info, client_metadata, request)
        registration_info = self.generate_client_registration_info(client, request)
        if registration_info:
            body.update(registration_info)
        return 201, body, default_json_headers

    def extract_client_metadata(self, request):
        if not request.payload.data:
            raise InvalidRequestError()

        json_data = request.payload.data.copy()
        software_statement = json_data.pop("software_statement", None)
        if software_statement and self.software_statement_alg_values_supported:
            data = self.extract_software_statement(software_statement, request)
            json_data.update(data)

        client_metadata = {}
        server_metadata = self.get_server_metadata()
        for claims_class in self.claims_classes:
            options = (
                claims_class.get_claims_options(server_metadata)
                if hasattr(claims_class, "get_claims_options") and server_metadata
                else {}
            )
            claims = claims_class(json_data, {}, options, server_metadata)
            try:
                claims.validate()
            except JoseError as error:
                raise InvalidClientMetadataError(error.description) from error

            client_metadata.update(**claims.get_registered_claims())
        return client_metadata

    def extract_software_statement(self, software_statement, request):
        key = self.resolve_public_key(request)
        if not key:
            raise UnapprovedSoftwareStatementError()

        try:
            jwt = JsonWebToken(self.software_statement_alg_values_supported)
            claims = jwt.decode(software_statement, key)
            # there is no need to validate claims
            return claims
        except JoseError as exc:
            raise InvalidSoftwareStatementError() from exc

    def generate_client_info(self, request):
        # https://tools.ietf.org/html/rfc7591#section-3.2.1
        try:
            client_id = self.generate_client_id(request)
        except TypeError:  # pragma: no cover
            client_id = self.generate_client_id()
            deprecate(
                "generate_client_id takes a 'request' parameter. "
                "It will become mandatory in coming releases",
                version="1.8",
            )

        try:
            client_secret = self.generate_client_secret(request)
        except TypeError:  # pragma: no cover
            client_secret = self.generate_client_secret()
            deprecate(
                "generate_client_secret takes a 'request' parameter. "
                "It will become mandatory in coming releases",
                version="1.8",
            )

        client_id_issued_at = int(time.time())
        client_secret_expires_at = 0
        return dict(
            client_id=client_id,
            client_secret=client_secret,
            client_id_issued_at=client_id_issued_at,
            client_secret_expires_at=client_secret_expires_at,
        )

    def generate_client_registration_info(self, client, request):
        """Generate ```registration_client_uri`` and ``registration_access_token``
        for RFC7592. This method returns ``None`` by default. Developers MAY rewrite
        this method to return registration information.
        """
        return None

    def create_endpoint_request(self, request):
        return self.server.create_json_request(request)

    def generate_client_id(self, request):
        """Generate ``client_id`` value. Developers MAY rewrite this method
        to use their own way to generate ``client_id``.
        """
        return generate_token(42)

    def generate_client_secret(self, request):
        """Generate ``client_secret`` value. Developers MAY rewrite this method
        to use their own way to generate ``client_secret``.
        """
        return binascii.hexlify(os.urandom(24)).decode("ascii")

    def get_server_metadata(self):
        """Return server metadata which includes supported grant types,
        response types and etc.
        """
        raise NotImplementedError()

    def authenticate_token(self, request):
        """Authenticate current credential who is requesting to register a client.
        Developers MUST implement this method in subclass::

            def authenticate_token(self, request):
                auth = request.headers.get("Authorization")
                return get_token_by_auth(auth)

        :return: token instance
        """
        raise NotImplementedError()

    def resolve_public_key(self, request):
        """Resolve a public key for decoding ``software_statement``. If
        ``enable_software_statement=True``, developers MUST implement this
        method in subclass::

            def resolve_public_key(self, request):
                return get_public_key_from_user(request.credential)

        :return: JWK or Key string
        """
        raise NotImplementedError()

    def save_client(self, client_info, client_metadata, request):
        """Save client into database. Developers MUST implement this method
        in subclass::

            def save_client(self, client_info, client_metadata, request):
                client = OAuthClient(
                    client_id=client_info['client_id'],
                    client_secret=client_info['client_secret'],
                    ...
                )
                client.save()
                return client
        """
        raise NotImplementedError()
