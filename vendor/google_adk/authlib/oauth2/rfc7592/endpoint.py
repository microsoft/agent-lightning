from authlib.consts import default_json_headers
from authlib.jose import JoseError

from ..rfc6749 import AccessDeniedError
from ..rfc6749 import InvalidClientError
from ..rfc6749 import InvalidRequestError
from ..rfc6749 import UnauthorizedClientError
from ..rfc7591 import InvalidClientMetadataError
from ..rfc7591.claims import ClientMetadataClaims


class ClientConfigurationEndpoint:
    ENDPOINT_NAME = "client_configuration"

    def __init__(self, server=None, claims_classes=None):
        self.server = server
        self.claims_classes = claims_classes or [ClientMetadataClaims]

    def __call__(self, request):
        return self.create_configuration_response(request)

    def create_configuration_response(self, request):
        # This request is authenticated by the registration access token issued
        # to the client.
        token = self.authenticate_token(request)
        if not token:
            raise AccessDeniedError()

        request.credential = token

        client = self.authenticate_client(request)
        if not client:
            # If the client does not exist on this server, the server MUST respond
            # with HTTP 401 Unauthorized and the registration access token used to
            # make this request SHOULD be immediately revoked.
            self.revoke_access_token(request, token)
            raise InvalidClientError(
                status_code=401, description="The client does not exist on this server."
            )

        if not self.check_permission(client, request):
            # If the client does not have permission to read its record, the server
            # MUST return an HTTP 403 Forbidden.
            raise UnauthorizedClientError(
                status_code=403,
                description="The client does not have permission to read its record.",
            )

        request.client = client

        if request.method == "GET":
            return self.create_read_client_response(client, request)
        elif request.method == "DELETE":
            return self.create_delete_client_response(client, request)
        elif request.method == "PUT":
            return self.create_update_client_response(client, request)

    def create_endpoint_request(self, request):
        return self.server.create_json_request(request)

    def create_read_client_response(self, client, request):
        body = self.introspect_client(client)
        body.update(self.generate_client_registration_info(client, request))
        return 200, body, default_json_headers

    def create_delete_client_response(self, client, request):
        self.delete_client(client, request)
        headers = [
            ("Cache-Control", "no-store"),
            ("Pragma", "no-cache"),
        ]
        return 204, "", headers

    def create_update_client_response(self, client, request):
        # The updated client metadata fields request MUST NOT include the
        # 'registration_access_token', 'registration_client_uri',
        # 'client_secret_expires_at', or 'client_id_issued_at' fields
        must_not_include = (
            "registration_access_token",
            "registration_client_uri",
            "client_secret_expires_at",
            "client_id_issued_at",
        )
        for k in must_not_include:
            if k in request.payload.data:
                raise InvalidRequestError()

        # The client MUST include its 'client_id' field in the request
        client_id = request.payload.data.get("client_id")
        if not client_id:
            raise InvalidRequestError()
        if client_id != client.get_client_id():
            raise InvalidRequestError()

        # If the client includes the 'client_secret' field in the request,
        # the value of this field MUST match the currently issued client
        # secret for that client.
        if "client_secret" in request.payload.data:
            if not client.check_client_secret(request.payload.data["client_secret"]):
                raise InvalidRequestError()

        client_metadata = self.extract_client_metadata(request)
        client = self.update_client(client, client_metadata, request)
        return self.create_read_client_response(client, request)

    def extract_client_metadata(self, request):
        json_data = request.payload.data.copy()
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

    def introspect_client(self, client):
        return {**client.client_info, **client.client_metadata}

    def generate_client_registration_info(self, client, request):
        """Generate ```registration_client_uri`` and ``registration_access_token``
        for RFC7592. By default this method returns the values sent in the current
        request. Developers MUST rewrite this method to return different registration
        information.::

            def generate_client_registration_info(self, client, request):{
                access_token = request.headers['Authorization'].split(' ')[1]
                return {
                    'registration_client_uri': request.uri,
                    'registration_access_token': access_token,
                }

        :param client: the instance of OAuth client
        :param request: formatted request instance
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

    def authenticate_client(self, request):
        """Read a client from the request payload.
        Developers MUST implement this method in subclass::

            def authenticate_client(self, request):
                client_id = request.payload.data.get("client_id")
                return Client.get(client_id=client_id)

        :return: client instance
        """
        raise NotImplementedError()

    def revoke_access_token(self, token, request):
        """Revoke a token access in case an invalid client has been requested.
        Developers MUST implement this method in subclass::

            def revoke_access_token(self, token, request):
                token.revoked = True
                token.save()

        """
        raise NotImplementedError()

    def check_permission(self, client, request):
        """Checks whether the current client is allowed to be accessed, edited
        or deleted. Developers MUST implement it in subclass, e.g.::

            def check_permission(self, client, request):
                return client.editable

        :return: boolean
        """
        raise NotImplementedError()

    def delete_client(self, client, request):
        """Delete authorization code from database or cache. Developers MUST
        implement it in subclass, e.g.::

            def delete_client(self, client, request):
                client.delete()

        :param client: the instance of OAuth client
        :param request: formatted request instance
        """
        raise NotImplementedError()

    def update_client(self, client, client_metadata, request):
        """Update the client in the database. Developers MUST implement this method
        in subclass::

            def update_client(self, client, client_metadata, request):
                client.set_client_metadata(
                    {**client.client_metadata, **client_metadata}
                )
                client.save()
                return client

        :param client: the instance of OAuth client
        :param client_metadata: a dict of the client claims to update
        :param request: formatted request instance
        :return: client instance
        """
        raise NotImplementedError()

    def get_server_metadata(self):
        """Return server metadata which includes supported grant types,
        response types and etc.
        """
        raise NotImplementedError()
