from authlib.jose import BaseClaims
from authlib.jose.errors import InvalidClaimError


class ClientMetadataClaims(BaseClaims):
    """Additional client metadata can be used with :ref:`specs/rfc7591` and :ref:`specs/rfc7592` endpoints.

    This can be used with::

        server.register_endpoint(
            ClientRegistrationEndpoint(
                claims_classes=[
                    rfc7591.ClientMetadataClaims,
                    rfc9101.ClientMetadataClaims,
                ]
            )
        )

        server.register_endpoint(
            ClientRegistrationEndpoint(
                claims_classes=[
                    rfc7591.ClientMetadataClaims,
                    rfc9101.ClientMetadataClaims,
                ]
            )
        )

    """

    REGISTERED_CLAIMS = [
        "require_signed_request_object",
    ]

    def validate(self):
        self._validate_essential_claims()
        self.validate_require_signed_request_object()

    def validate_require_signed_request_object(self):
        self.setdefault("require_signed_request_object", False)

        if not isinstance(self["require_signed_request_object"], bool):
            raise InvalidClaimError("require_signed_request_object")

        self._validate_claim_value("require_signed_request_object")
