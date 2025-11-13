from authlib.common.security import generate_token
from authlib.common.urls import add_params_to_uri
from authlib.consts import default_json_headers


class DeviceAuthorizationEndpoint:
    """This OAuth 2.0 [RFC6749] protocol extension enables OAuth clients to
    request user authorization from applications on devices that have
    limited input capabilities or lack a suitable browser.  Such devices
    include smart TVs, media consoles, picture frames, and printers,
    which lack an easy input method or a suitable browser required for
    traditional OAuth interactions. Here is the authorization flow::

        +----------+                                +----------------+
        |          |>---(A)-- Client Identifier --->|                |
        |          |                                |                |
        |          |<---(B)-- Device Code,      ---<|                |
        |          |          User Code,            |                |
        |  Device  |          & Verification URI    |                |
        |  Client  |                                |                |
        |          |  [polling]                     |                |
        |          |>---(E)-- Device Code       --->|                |
        |          |          & Client Identifier   |                |
        |          |                                |  Authorization |
        |          |<---(F)-- Access Token      ---<|     Server     |
        +----------+   (& Optional Refresh Token)   |                |
              v                                     |                |
              :                                     |                |
             (C) User Code & Verification URI       |                |
              :                                     |                |
              v                                     |                |
        +----------+                                |                |
        | End User |                                |                |
        |    at    |<---(D)-- End user reviews  --->|                |
        |  Browser |          authorization request |                |
        +----------+                                +----------------+

    This DeviceAuthorizationEndpoint is the implementation of step (A) and (B).

    (A) The client requests access from the authorization server and
        includes its client identifier in the request.

    (B) The authorization server issues a device code and an end-user
        code and provides the end-user verification URI.
    """

    ENDPOINT_NAME = "device_authorization"
    CLIENT_AUTH_METHODS = ["client_secret_basic", "client_secret_post", "none"]

    #: customize "user_code" type, string or digital
    USER_CODE_TYPE = "string"

    #: The lifetime in seconds of the "device_code" and "user_code"
    EXPIRES_IN = 1800

    #: The minimum amount of time in seconds that the client SHOULD
    #: wait between polling requests to the token endpoint.
    INTERVAL = 5

    def __init__(self, server):
        self.server = server

    def __call__(self, request):
        # make it callable for authorization server
        # ``create_endpoint_response``
        return self.create_endpoint_response(request)

    def create_endpoint_request(self, request):
        return self.server.create_oauth2_request(request)

    def authenticate_client(self, request):
        """client_id is REQUIRED **if the client is not** authenticating with the
        authorization server as described in Section 3.2.1. of [RFC6749].

        This means the endpoint support "none" authentication method. In this case,
        this endpoint's auth methods are:

        - client_secret_basic
        - client_secret_post
        - none

        Developers change the value of ``CLIENT_AUTH_METHODS`` in subclass. For
        instance::

            class MyDeviceAuthorizationEndpoint(DeviceAuthorizationEndpoint):
                # only support ``client_secret_basic`` auth method
                CLIENT_AUTH_METHODS = ["client_secret_basic"]
        """
        client = self.server.authenticate_client(
            request, self.CLIENT_AUTH_METHODS, self.ENDPOINT_NAME
        )
        request.client = client
        return client

    def create_endpoint_response(self, request):
        # https://tools.ietf.org/html/rfc8628#section-3.1

        self.authenticate_client(request)
        self.server.validate_requested_scope(request.payload.scope)

        device_code = self.generate_device_code()
        user_code = self.generate_user_code()
        verification_uri = self.get_verification_uri()
        verification_uri_complete = add_params_to_uri(
            verification_uri, [("user_code", user_code)]
        )

        data = {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": verification_uri,
            "verification_uri_complete": verification_uri_complete,
            "expires_in": self.EXPIRES_IN,
            "interval": self.INTERVAL,
        }

        self.save_device_credential(
            request.payload.client_id, request.payload.scope, data
        )
        return 200, data, default_json_headers

    def generate_user_code(self):
        """A method to generate ``user_code`` value for device authorization
        endpoint. This method will generate a random string like MQNA-JPOZ.
        Developers can rewrite this  method to create their own ``user_code``.
        """
        # https://tools.ietf.org/html/rfc8628#section-6.1
        if self.USER_CODE_TYPE == "digital":
            return create_digital_user_code()
        return create_string_user_code()

    def generate_device_code(self):
        """A method to generate ``device_code`` value for device authorization
        endpoint. This method will generate a random string of 42 characters.
        Developers can rewrite this method to create their own ``device_code``.
        """
        return generate_token(42)

    def get_verification_uri(self):
        """Define the ``verification_uri`` of device authorization endpoint.
        Developers MUST implement this method in subclass::

            def get_verification_uri(self):
                return "https://your-company.com/active"
        """
        raise NotImplementedError()

    def save_device_credential(self, client_id, scope, data):
        """Save device token into database for later use. Developers MUST
        implement this method in subclass::

            def save_device_credential(self, client_id, scope, data):
                item = DeviceCredential(client_id=client_id, scope=scope, **data)
                item.save()
        """
        raise NotImplementedError()


def create_string_user_code():
    base = "BCDFGHJKLMNPQRSTVWXZ"
    return "-".join([generate_token(4, base), generate_token(4, base)])


def create_digital_user_code():
    base = "0123456789"
    return "-".join(
        [
            generate_token(3, base),
            generate_token(3, base),
            generate_token(3, base),
        ]
    )
