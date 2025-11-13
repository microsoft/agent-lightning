from typing import Optional

from authlib.common.urls import add_params_to_uri
from authlib.deprecate import deprecate
from authlib.oauth2.rfc6749.grants import BaseGrant


class IssuerParameter:
    def __call__(self, authorization_server):
        if isinstance(authorization_server, BaseGrant):
            deprecate(
                "IssueParameter should be used as an authorization server extension with 'authorization_server.register_extension(IssueParameter())'.",
                version="1.8",
            )
            authorization_server.register_hook(
                "after_authorization_response",
                self.add_issuer_parameter,
            )

        else:
            authorization_server.register_hook(
                "after_create_authorization_response",
                self.add_issuer_parameter,
            )

    def add_issuer_parameter(self, authorization_server, response):
        if self.get_issuer() and response.location:
            # RFC9207 §2
            # In authorization responses to the client, including error responses,
            # an authorization server supporting this specification MUST indicate
            # its identity by including the iss parameter in the response.

            new_location = add_params_to_uri(
                response.location, {"iss": self.get_issuer()}
            )
            response.location = new_location

    def get_issuer(self) -> Optional[str]:
        """Return the issuer URL.
        Developers MAY implement this method if they want to support :rfc:`RFC9207 <9207>`::

            def get_issuer(self) -> str:
                return "https://auth.example.org"
        """
        return None
