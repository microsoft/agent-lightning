from authlib.common.urls import is_valid_url
from authlib.jose import BaseClaims
from authlib.jose.errors import InvalidClaimError


class ClientMetadataClaims(BaseClaims):
    REGISTERED_CLAIMS = [
        "token_endpoint_auth_signing_alg",
        "application_type",
        "sector_identifier_uri",
        "subject_type",
        "id_token_signed_response_alg",
        "id_token_encrypted_response_alg",
        "id_token_encrypted_response_enc",
        "userinfo_signed_response_alg",
        "userinfo_encrypted_response_alg",
        "userinfo_encrypted_response_enc",
        "default_max_age",
        "require_auth_time",
        "default_acr_values",
        "initiate_login_uri",
        "request_object_signing_alg",
        "request_object_encryption_alg",
        "request_object_encryption_enc",
        "request_uris",
    ]

    def validate(self):
        self._validate_essential_claims()
        self.validate_token_endpoint_auth_signing_alg()
        self.validate_application_type()
        self.validate_sector_identifier_uri()
        self.validate_subject_type()
        self.validate_id_token_signed_response_alg()
        self.validate_id_token_encrypted_response_alg()
        self.validate_id_token_encrypted_response_enc()
        self.validate_userinfo_signed_response_alg()
        self.validate_userinfo_encrypted_response_alg()
        self.validate_userinfo_encrypted_response_enc()
        self.validate_default_max_age()
        self.validate_require_auth_time()
        self.validate_default_acr_values()
        self.validate_initiate_login_uri()
        self.validate_request_object_signing_alg()
        self.validate_request_object_encryption_alg()
        self.validate_request_object_encryption_enc()
        self.validate_request_uris()

    def _validate_uri(self, key):
        uri = self.get(key)
        uris = uri if isinstance(uri, list) else [uri]
        for uri in uris:
            if uri and not is_valid_url(uri):
                raise InvalidClaimError(key)

    @classmethod
    def get_claims_options(self, metadata):
        """Generate claims options validation from Authorization Server metadata."""
        options = {}

        if acr_values_supported := metadata.get("acr_values_supported"):

            def _validate_default_acr_values(claims, value):
                return not value or set(value).issubset(set(acr_values_supported))

            options["default_acr_values"] = {"validate": _validate_default_acr_values}

        values_mapping = {
            "token_endpoint_auth_signing_alg_values_supported": "token_endpoint_auth_signing_alg",
            "subject_types_supported": "subject_type",
            "id_token_signing_alg_values_supported": "id_token_signed_response_alg",
            "id_token_encryption_alg_values_supported": "id_token_encrypted_response_alg",
            "id_token_encryption_enc_values_supported": "id_token_encrypted_response_enc",
            "userinfo_signing_alg_values_supported": "userinfo_signed_response_alg",
            "userinfo_encryption_alg_values_supported": "userinfo_encrypted_response_alg",
            "userinfo_encryption_enc_values_supported": "userinfo_encrypted_response_enc",
            "request_object_signing_alg_values_supported": "request_object_signing_alg",
            "request_object_encryption_alg_values_supported": "request_object_encryption_alg",
            "request_object_encryption_enc_values_supported": "request_object_encryption_enc",
        }

        def make_validator(metadata_claim_values):
            def _validate(claims, value):
                return not value or value in metadata_claim_values

            return _validate

        for metadata_claim_name, request_claim_name in values_mapping.items():
            if metadata_claim_values := metadata.get(metadata_claim_name):
                options[request_claim_name] = {
                    "validate": make_validator(metadata_claim_values)
                }

        return options

    def validate_token_endpoint_auth_signing_alg(self):
        """JWS [JWS] alg algorithm [JWA] that MUST be used for signing the JWT [JWT]
        used to authenticate the Client at the Token Endpoint for the private_key_jwt
        and client_secret_jwt authentication methods.

        All Token Requests using these authentication methods from this Client MUST be
        rejected, if the JWT is not signed with this algorithm. Servers SHOULD support
        RS256. The value none MUST NOT be used. The default, if omitted, is that any
        algorithm supported by the OP and the RP MAY be used.
        """
        if self.get("token_endpoint_auth_signing_alg") == "none":
            raise InvalidClaimError("token_endpoint_auth_signing_alg")

        self._validate_claim_value("token_endpoint_auth_signing_alg")

    def validate_application_type(self):
        """Kind of the application.

        The default, if omitted, is web. The defined values are native or web. Web
        Clients using the OAuth Implicit Grant Type MUST only register URLs using the
        https scheme as redirect_uris; they MUST NOT use localhost as the hostname.
        Native Clients MUST only register redirect_uris using custom URI schemes or
        loopback URLs using the http scheme; loopback URLs use localhost or the IP
        loopback literals 127.0.0.1 or [::1] as the hostname. Authorization Servers MAY
        place additional constraints on Native Clients. Authorization Servers MAY
        reject Redirection URI values using the http scheme, other than the loopback
        case for Native Clients. The Authorization Server MUST verify that all the
        registered redirect_uris conform to these constraints. This prevents sharing a
        Client ID across different types of Clients.
        """
        self.setdefault("application_type", "web")
        if self.get("application_type") not in ("web", "native"):
            raise InvalidClaimError("application_type")

        self._validate_claim_value("application_type")

    def validate_sector_identifier_uri(self):
        """URL using the https scheme to be used in calculating Pseudonymous Identifiers
        by the OP.

        The URL references a file with a single JSON array of redirect_uri values.
        Please see Section 5. Providers that use pairwise sub (subject) values SHOULD
        utilize the sector_identifier_uri value provided in the Subject Identifier
        calculation for pairwise identifiers.
        """
        self._validate_uri("sector_identifier_uri")

    def validate_subject_type(self):
        """subject_type requested for responses to this Client.

        The subject_types_supported discovery parameter contains a list of the supported
        subject_type values for the OP. Valid types include pairwise and public.
        """
        self._validate_claim_value("subject_type")

    def validate_id_token_signed_response_alg(self):
        """JWS alg algorithm [JWA] REQUIRED for signing the ID Token issued to this
        Client.

        The value none MUST NOT be used as the ID Token alg value unless the Client uses
        only Response Types that return no ID Token from the Authorization Endpoint
        (such as when only using the Authorization Code Flow). The default, if omitted,
        is RS256. The public key for validating the signature is provided by retrieving
        the JWK Set referenced by the jwks_uri element from OpenID Connect Discovery 1.0
        [OpenID.Discovery].
        """
        if self.get(
            "id_token_signed_response_alg"
        ) == "none" and "id_token" in self.get("response_type", ""):
            raise InvalidClaimError("id_token_signed_response_alg")

        self.setdefault("id_token_signed_response_alg", "RS256")
        self._validate_claim_value("id_token_signed_response_alg")

    def validate_id_token_encrypted_response_alg(self):
        """JWE alg algorithm [JWA] REQUIRED for encrypting the ID Token issued to this
        Client.

        If this is requested, the response will be signed then encrypted, with the
        result being a Nested JWT, as defined in [JWT]. The default, if omitted, is that
        no encryption is performed.
        """
        self._validate_claim_value("id_token_encrypted_response_alg")

    def validate_id_token_encrypted_response_enc(self):
        """JWE enc algorithm [JWA] REQUIRED for encrypting the ID Token issued to this
        Client.

        If id_token_encrypted_response_alg is specified, the default
        id_token_encrypted_response_enc value is A128CBC-HS256. When
        id_token_encrypted_response_enc is included, id_token_encrypted_response_alg
        MUST also be provided.
        """
        if self.get("id_token_encrypted_response_enc") and not self.get(
            "id_token_encrypted_response_alg"
        ):
            raise InvalidClaimError("id_token_encrypted_response_enc")

        if self.get("id_token_encrypted_response_alg"):
            self.setdefault("id_token_encrypted_response_enc", "A128CBC-HS256")

        self._validate_claim_value("id_token_encrypted_response_enc")

    def validate_userinfo_signed_response_alg(self):
        """JWS alg algorithm [JWA] REQUIRED for signing UserInfo Responses.

        If this is specified, the response will be JWT [JWT] serialized, and signed
        using JWS. The default, if omitted, is for the UserInfo Response to return the
        Claims as a UTF-8 [RFC3629] encoded JSON object using the application/json
        content-type.
        """
        self._validate_claim_value("userinfo_signed_response_alg")

    def validate_userinfo_encrypted_response_alg(self):
        """JWE [JWE] alg algorithm [JWA] REQUIRED for encrypting UserInfo Responses.

        If both signing and encryption are requested, the response will be signed then
        encrypted, with the result being a Nested JWT, as defined in [JWT]. The default,
        if omitted, is that no encryption is performed.
        """
        self._validate_claim_value("userinfo_encrypted_response_alg")

    def validate_userinfo_encrypted_response_enc(self):
        """JWE enc algorithm [JWA] REQUIRED for encrypting UserInfo Responses.

        If userinfo_encrypted_response_alg is specified, the default
        userinfo_encrypted_response_enc value is A128CBC-HS256. When
        userinfo_encrypted_response_enc is included, userinfo_encrypted_response_alg
        MUST also be provided.
        """
        if self.get("userinfo_encrypted_response_enc") and not self.get(
            "userinfo_encrypted_response_alg"
        ):
            raise InvalidClaimError("userinfo_encrypted_response_enc")

        if self.get("userinfo_encrypted_response_alg"):
            self.setdefault("userinfo_encrypted_response_enc", "A128CBC-HS256")

        self._validate_claim_value("userinfo_encrypted_response_enc")

    def validate_default_max_age(self):
        """Default Maximum Authentication Age.

        Specifies that the End-User MUST be actively authenticated if the End-User was
        authenticated longer ago than the specified number of seconds. The max_age
        request parameter overrides this default value. If omitted, no default Maximum
        Authentication Age is specified.
        """
        if self.get("default_max_age") is not None and not isinstance(
            self["default_max_age"], (int, float)
        ):
            raise InvalidClaimError("default_max_age")

        self._validate_claim_value("default_max_age")

    def validate_require_auth_time(self):
        """Boolean value specifying whether the auth_time Claim in the ID Token is
        REQUIRED.

        It is REQUIRED when the value is true. (If this is false, the auth_time Claim
        can still be dynamically requested as an individual Claim for the ID Token using
        the claims request parameter described in Section 5.5.1 of OpenID Connect Core
        1.0 [OpenID.Core].) If omitted, the default value is false.
        """
        self.setdefault("require_auth_time", False)
        if self.get("require_auth_time") is not None and not isinstance(
            self["require_auth_time"], bool
        ):
            raise InvalidClaimError("require_auth_time")

        self._validate_claim_value("require_auth_time")

    def validate_default_acr_values(self):
        """Default requested Authentication Context Class Reference values.

        Array of strings that specifies the default acr values that the OP is being
        requested to use for processing requests from this Client, with the values
        appearing in order of preference. The Authentication Context Class satisfied by
        the authentication performed is returned as the acr Claim Value in the issued ID
        Token. The acr Claim is requested as a Voluntary Claim by this parameter. The
        acr_values_supported discovery element contains a list of the supported acr
        values supported by the OP. Values specified in the acr_values request parameter
        or an individual acr Claim request override these default values.
        """
        self._validate_claim_value("default_acr_values")

    def validate_initiate_login_uri(self):
        """RI using the https scheme that a third party can use to initiate a login by
        the RP, as specified in Section 4 of OpenID Connect Core 1.0 [OpenID.Core].

        The URI MUST accept requests via both GET and POST. The Client MUST understand
        the login_hint and iss parameters and SHOULD support the target_link_uri
        parameter.
        """
        self._validate_uri("initiate_login_uri")

    def validate_request_object_signing_alg(self):
        """JWS [JWS] alg algorithm [JWA] that MUST be used for signing Request Objects
        sent to the OP.

        All Request Objects from this Client MUST be rejected, if not signed with this
        algorithm. Request Objects are described in Section 6.1 of OpenID Connect Core
        1.0 [OpenID.Core]. This algorithm MUST be used both when the Request Object is
        passed by value (using the request parameter) and when it is passed by reference
        (using the request_uri parameter). Servers SHOULD support RS256. The value none
        MAY be used. The default, if omitted, is that any algorithm supported by the OP
        and the RP MAY be used.
        """
        self._validate_claim_value("request_object_signing_alg")

    def validate_request_object_encryption_alg(self):
        """JWE [JWE] alg algorithm [JWA] the RP is declaring that it may use for
        encrypting Request Objects sent to the OP.

        This parameter SHOULD be included when symmetric encryption will be used, since
        this signals to the OP that a client_secret value needs to be returned from
        which the symmetric key will be derived, that might not otherwise be returned.
        The RP MAY still use other supported encryption algorithms or send unencrypted
        Request Objects, even when this parameter is present. If both signing and
        encryption are requested, the Request Object will be signed then encrypted, with
        the result being a Nested JWT, as defined in [JWT]. The default, if omitted, is
        that the RP is not declaring whether it might encrypt any Request Objects.
        """
        self._validate_claim_value("request_object_encryption_alg")

    def validate_request_object_encryption_enc(self):
        """JWE enc algorithm [JWA] the RP is declaring that it may use for encrypting
        Request Objects sent to the OP.

        If request_object_encryption_alg is specified, the default
        request_object_encryption_enc value is A128CBC-HS256. When
        request_object_encryption_enc is included, request_object_encryption_alg MUST
        also be provided.
        """
        if self.get("request_object_encryption_enc") and not self.get(
            "request_object_encryption_alg"
        ):
            raise InvalidClaimError("request_object_encryption_enc")

        if self.get("request_object_encryption_alg"):
            self.setdefault("request_object_encryption_enc", "A128CBC-HS256")

        self._validate_claim_value("request_object_encryption_enc")

    def validate_request_uris(self):
        """Array of request_uri values that are pre-registered by the RP for use at the
        OP.

        These URLs MUST use the https scheme unless the target Request Object is signed
        in a way that is verifiable by the OP. Servers MAY cache the contents of the
        files referenced by these URIs and not retrieve them at the time they are used
        in a request. OPs can require that request_uri values used be pre-registered
        with the require_request_uri_registration discovery parameter. If the contents
        of the request file could ever change, these URI values SHOULD include the
        base64url-encoded SHA-256 hash value of the file contents referenced by the URI
        as the value of the URI fragment. If the fragment value used for a URI changes,
        that signals the server that its cached value for that URI with the old fragment
        value is no longer valid.
        """
        self._validate_uri("request_uris")
