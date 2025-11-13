class TokenEndpoint:
    #: Endpoint name to be registered
    ENDPOINT_NAME = None
    #: Supported token types
    SUPPORTED_TOKEN_TYPES = ("access_token", "refresh_token")
    #: Allowed client authenticate methods
    CLIENT_AUTH_METHODS = ["client_secret_basic"]

    def __init__(self, server):
        self.server = server

    def __call__(self, request):
        # make it callable for authorization server
        # ``create_endpoint_response``
        return self.create_endpoint_response(request)

    def create_endpoint_request(self, request):
        return self.server.create_oauth2_request(request)

    def authenticate_endpoint_client(self, request):
        """Authentication client for endpoint with ``CLIENT_AUTH_METHODS``."""
        client = self.server.authenticate_client(
            request, self.CLIENT_AUTH_METHODS, self.ENDPOINT_NAME
        )
        request.client = client
        return client

    def authenticate_token(self, request, client):
        raise NotImplementedError()

    def create_endpoint_response(self, request):
        raise NotImplementedError()
