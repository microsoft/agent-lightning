from authlib.common.errors import ContinueIteration
from authlib.deprecate import deprecate

from .authenticate_client import ClientAuthentication
from .errors import InvalidScopeError
from .errors import OAuth2Error
from .errors import UnsupportedGrantTypeError
from .errors import UnsupportedResponseTypeError
from .hooks import Hookable
from .hooks import hooked
from .requests import JsonRequest
from .requests import OAuth2Request
from .util import scope_to_list


class AuthorizationServer(Hookable):
    """Authorization server that handles Authorization Endpoint and Token
    Endpoint.

    :param scopes_supported: A list of supported scopes by this authorization server.
    """

    def __init__(self, scopes_supported=None):
        super().__init__()
        self.scopes_supported = scopes_supported
        self._token_generators = {}
        self._client_auth = None
        self._authorization_grants = []
        self._token_grants = []
        self._endpoints = {}
        self._extensions = []

    def query_client(self, client_id):
        """Query OAuth client by client_id. The client model class MUST
        implement the methods described by
        :class:`~authlib.oauth2.rfc6749.ClientMixin`.
        """
        raise NotImplementedError()

    def save_token(self, token, request):
        """Define function to save the generated token into database."""
        raise NotImplementedError()

    def generate_token(
        self,
        grant_type,
        client,
        user=None,
        scope=None,
        expires_in=None,
        include_refresh_token=True,
    ):
        """Generate the token dict.

        :param grant_type: current requested grant_type.
        :param client: the client that making the request.
        :param user: current authorized user.
        :param expires_in: if provided, use this value as expires_in.
        :param scope: current requested scope.
        :param include_refresh_token: should refresh_token be included.
        :return: Token dict
        """
        # generator for a specified grant type
        func = self._token_generators.get(grant_type)
        if not func:
            # default generator for all grant types
            func = self._token_generators.get("default")
        if not func:
            raise RuntimeError("No configured token generator")

        return func(
            grant_type=grant_type,
            client=client,
            user=user,
            scope=scope,
            expires_in=expires_in,
            include_refresh_token=include_refresh_token,
        )

    def register_token_generator(self, grant_type, func):
        """Register a function as token generator for the given ``grant_type``.
        Developers MUST register a default token generator with a special
        ``grant_type=default``::

            def generate_bearer_token(
                grant_type,
                client,
                user=None,
                scope=None,
                expires_in=None,
                include_refresh_token=True,
            ):
                token = {"token_type": "Bearer", "access_token": ...}
                if include_refresh_token:
                    token["refresh_token"] = ...
                ...
                return token


            authorization_server.register_token_generator(
                "default", generate_bearer_token
            )

        If you register a generator for a certain grant type, that generator will only works
        for the given grant type::

            authorization_server.register_token_generator(
                "client_credentials",
                generate_bearer_token,
            )

        :param grant_type: string name of the grant type
        :param func: a function to generate token
        """
        self._token_generators[grant_type] = func

    def authenticate_client(self, request, methods, endpoint="token"):
        """Authenticate client via HTTP request information with the given
        methods, such as ``client_secret_basic``, ``client_secret_post``.
        """
        if self._client_auth is None and self.query_client:
            self._client_auth = ClientAuthentication(self.query_client)
        return self._client_auth(request, methods, endpoint)

    def register_client_auth_method(self, method, func):
        """Add more client auth method. The default methods are:

        * none: The client is a public client and does not have a client secret
        * client_secret_post: The client uses the HTTP POST parameters
        * client_secret_basic: The client uses HTTP Basic

        :param method: Name of the Auth method
        :param func: Function to authenticate the client

        The auth method accept two parameters: ``query_client`` and ``request``,
        an example for this method::

            def authenticate_client_via_custom(query_client, request):
                client_id = request.headers["X-Client-Id"]
                client = query_client(client_id)
                do_some_validation(client)
                return client


            authorization_server.register_client_auth_method(
                "custom", authenticate_client_via_custom
            )
        """
        if self._client_auth is None and self.query_client:
            self._client_auth = ClientAuthentication(self.query_client)

        self._client_auth.register(method, func)

    def register_extension(self, extension):
        self._extensions.append(extension(self))

    def get_error_uri(self, request, error):
        """Return a URI for the given error, framework may implement this method."""
        return None

    def send_signal(self, name, *args, **kwargs):
        """Framework integration can re-implement this method to support
        signal system.
        """
        raise NotImplementedError()

    def create_oauth2_request(self, request) -> OAuth2Request:
        """This method MUST be implemented in framework integrations. It is
        used to create an OAuth2Request instance.

        :param request: the "request" instance in framework
        :return: OAuth2Request instance
        """
        raise NotImplementedError()

    def create_json_request(self, request) -> JsonRequest:
        """This method MUST be implemented in framework integrations. It is
        used to create an HttpRequest instance.

        :param request: the "request" instance in framework
        :return: HttpRequest instance
        """
        raise NotImplementedError()

    def handle_response(self, status, body, headers):
        """Return HTTP response. Framework MUST implement this function."""
        raise NotImplementedError()

    def validate_requested_scope(self, scope):
        """Validate if requested scope is supported by Authorization Server.
        Developers CAN re-write this method to meet your needs.
        """
        if scope and self.scopes_supported:
            scopes = set(scope_to_list(scope))
            if not set(self.scopes_supported).issuperset(scopes):
                raise InvalidScopeError()

    def register_grant(self, grant_cls, extensions=None):
        """Register a grant class into the endpoint registry. Developers
        can implement the grants in ``authlib.oauth2.rfc6749.grants`` and
        register with this method::

            class AuthorizationCodeGrant(grants.AuthorizationCodeGrant):
                def authenticate_user(self, credential):
                    # ...

            authorization_server.register_grant(AuthorizationCodeGrant)

        :param grant_cls: a grant class.
        :param extensions: extensions for the grant class.
        """
        if hasattr(grant_cls, "check_authorization_endpoint"):
            self._authorization_grants.append((grant_cls, extensions))
        if hasattr(grant_cls, "check_token_endpoint"):
            self._token_grants.append((grant_cls, extensions))

    def register_endpoint(self, endpoint):
        """Add extra endpoint to authorization server. e.g.
        RevocationEndpoint::

            authorization_server.register_endpoint(RevocationEndpoint)

        :param endpoint_cls: A endpoint class or instance.
        """
        if isinstance(endpoint, type):
            endpoint = endpoint(self)
        else:
            endpoint.server = self

        endpoints = self._endpoints.setdefault(endpoint.ENDPOINT_NAME, [])
        endpoints.append(endpoint)

    @hooked
    def get_authorization_grant(self, request):
        """Find the authorization grant for current request.

        :param request: OAuth2Request instance.
        :return: grant instance
        """
        for grant_cls, extensions in self._authorization_grants:
            if grant_cls.check_authorization_endpoint(request):
                return _create_grant(grant_cls, extensions, request, self)

        raise UnsupportedResponseTypeError(
            f"The response type '{request.payload.response_type}' is not supported by the server.",
            request.payload.response_type,
            redirect_uri=request.payload.redirect_uri,
        )

    def get_consent_grant(self, request=None, end_user=None):
        """Validate current HTTP request for authorization page. This page
        is designed for resource owner to grant or deny the authorization.
        """
        request = self.create_oauth2_request(request)

        try:
            request.user = end_user

            grant = self.get_authorization_grant(request)
            grant.validate_no_multiple_request_parameter(request)
            grant.validate_consent_request()

        except OAuth2Error as error:
            # REQUIRED if a "state" parameter was present in the client
            # authorization request.  The exact value received from the
            # client.
            error.state = request.payload.state
            raise
        return grant

    def get_token_grant(self, request):
        """Find the token grant for current request.

        :param request: OAuth2Request instance.
        :return: grant instance
        """
        for grant_cls, extensions in self._token_grants:
            if grant_cls.check_token_endpoint(request):
                return _create_grant(grant_cls, extensions, request, self)
        raise UnsupportedGrantTypeError(request.payload.grant_type)

    def create_endpoint_response(self, name, request=None):
        """Validate endpoint request and create endpoint response.

        :param name: Endpoint name
        :param request: HTTP request instance.
        :return: Response
        """
        if name not in self._endpoints:
            raise RuntimeError(f"There is no '{name}' endpoint.")

        endpoints = self._endpoints[name]
        for endpoint in endpoints:
            request = endpoint.create_endpoint_request(request)
            try:
                return self.handle_response(*endpoint(request))
            except ContinueIteration:
                continue
            except OAuth2Error as error:
                return self.handle_error_response(request, error)

    @hooked
    def create_authorization_response(self, request=None, grant_user=None, grant=None):
        """Validate authorization request and create authorization response.

        :param request: HTTP request instance.
        :param grant_user: if granted, it is resource owner. If denied,
            it is None.
        :returns: Response
        """
        if not isinstance(request, OAuth2Request):
            request = self.create_oauth2_request(request)

        if not grant:
            deprecate("The 'grant' parameter will become mandatory.", version="1.8")
            try:
                grant = self.get_authorization_grant(request)
            except UnsupportedResponseTypeError as error:
                error.state = request.payload.state
                return self.handle_error_response(request, error)

        try:
            redirect_uri = grant.validate_authorization_request()
            args = grant.create_authorization_response(redirect_uri, grant_user)
            response = self.handle_response(*args)
        except OAuth2Error as error:
            error.state = request.payload.state
            response = self.handle_error_response(request, error)

        grant.execute_hook("after_authorization_response", response)
        return response

    def create_token_response(self, request=None):
        """Validate token request and create token response.

        :param request: HTTP request instance
        """
        request = self.create_oauth2_request(request)
        try:
            grant = self.get_token_grant(request)
        except UnsupportedGrantTypeError as error:
            return self.handle_error_response(request, error)

        try:
            grant.validate_token_request()
            args = grant.create_token_response()
            return self.handle_response(*args)
        except OAuth2Error as error:
            return self.handle_error_response(request, error)

    def handle_error_response(self, request, error):
        return self.handle_response(*error(self.get_error_uri(request, error)))


def _create_grant(grant_cls, extensions, request, server):
    grant = grant_cls(request, server)
    if extensions:
        for ext in extensions:
            ext(grant)
    return grant
