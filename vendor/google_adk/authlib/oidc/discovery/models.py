from authlib.oauth2.rfc8414 import AuthorizationServerMetadata
from authlib.oauth2.rfc8414.models import validate_array_value


class OpenIDProviderMetadata(AuthorizationServerMetadata):
    REGISTRY_KEYS = [
        "issuer",
        "authorization_endpoint",
        "token_endpoint",
        "jwks_uri",
        "registration_endpoint",
        "scopes_supported",
        "response_types_supported",
        "response_modes_supported",
        "grant_types_supported",
        "token_endpoint_auth_methods_supported",
        "service_documentation",
        "ui_locales_supported",
        "op_policy_uri",
        "op_tos_uri",
        # added by OpenID
        "token_endpoint_auth_signing_alg_values_supported",
        "acr_values_supported",
        "subject_types_supported",
        "id_token_signing_alg_values_supported",
        "id_token_encryption_alg_values_supported",
        "id_token_encryption_enc_values_supported",
        "userinfo_signing_alg_values_supported",
        "userinfo_encryption_alg_values_supported",
        "userinfo_encryption_enc_values_supported",
        "request_object_signing_alg_values_supported",
        "request_object_encryption_alg_values_supported",
        "request_object_encryption_enc_values_supported",
        "display_values_supported",
        "claim_types_supported",
        "claims_supported",
        "claims_locales_supported",
        "claims_parameter_supported",
        "request_parameter_supported",
        "request_uri_parameter_supported",
        "require_request_uri_registration",
        # not defined by OpenID
        # 'revocation_endpoint',
        # 'revocation_endpoint_auth_methods_supported',
        # 'revocation_endpoint_auth_signing_alg_values_supported',
        # 'introspection_endpoint',
        # 'introspection_endpoint_auth_methods_supported',
        # 'introspection_endpoint_auth_signing_alg_values_supported',
        # 'code_challenge_methods_supported',
    ]

    def validate_jwks_uri(self):
        # REQUIRED in OpenID Connect
        jwks_uri = self.get("jwks_uri")
        if jwks_uri is None:
            raise ValueError('"jwks_uri" is required')
        return super().validate_jwks_uri()

    def validate_acr_values_supported(self):
        """OPTIONAL. JSON array containing a list of the Authentication
        Context Class References that this OP supports.
        """
        validate_array_value(self, "acr_values_supported")

    def validate_subject_types_supported(self):
        """REQUIRED. JSON array containing a list of the Subject Identifier
        types that this OP supports. Valid types include pairwise and public.
        """
        # 1. REQUIRED
        values = self.get("subject_types_supported")
        if values is None:
            raise ValueError('"subject_types_supported" is required')

        # 2. JSON array
        if not isinstance(values, list):
            raise ValueError('"subject_types_supported" MUST be JSON array')

        # 3. Valid types include pairwise and public
        valid_types = {"pairwise", "public"}
        if not valid_types.issuperset(set(values)):
            raise ValueError('"subject_types_supported" contains invalid values')

    def validate_id_token_signing_alg_values_supported(self):
        """REQUIRED. JSON array containing a list of the JWS signing
        algorithms (alg values) supported by the OP for the ID Token to
        encode the Claims in a JWT [JWT]. The algorithm RS256 MUST be
        included. The value none MAY be supported, but MUST NOT be used
        unless the Response Type used returns no ID Token from the
        Authorization Endpoint (such as when using the Authorization
        Code Flow).
        """
        # 1. REQUIRED
        values = self.get("id_token_signing_alg_values_supported")
        if values is None:
            raise ValueError('"id_token_signing_alg_values_supported" is required')

        # 2. JSON array
        if not isinstance(values, list):
            raise ValueError(
                '"id_token_signing_alg_values_supported" MUST be JSON array'
            )

        # 3. The algorithm RS256 MUST be included
        if "RS256" not in values:
            raise ValueError(
                '"RS256" MUST be included in "id_token_signing_alg_values_supported"'
            )

    def validate_id_token_encryption_alg_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWE encryption
        algorithms (alg values) supported by the OP for the ID Token to
        encode the Claims in a JWT.
        """
        validate_array_value(self, "id_token_encryption_alg_values_supported")

    def validate_id_token_encryption_enc_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWE encryption
        algorithms (enc values) supported by the OP for the ID Token to
        encode the Claims in a JWT.
        """
        validate_array_value(self, "id_token_encryption_enc_values_supported")

    def validate_userinfo_signing_alg_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWS signing
        algorithms (alg values) [JWA] supported by the UserInfo Endpoint
        to encode the Claims in a JWT. The value none MAY be included.
        """
        validate_array_value(self, "userinfo_signing_alg_values_supported")

    def validate_userinfo_encryption_alg_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWE encryption
        algorithms (alg values) [JWA] supported by the UserInfo Endpoint
        to encode the Claims in a JWT.
        """
        validate_array_value(self, "userinfo_encryption_alg_values_supported")

    def validate_userinfo_encryption_enc_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWE encryption
        algorithms (enc values) [JWA] supported by the UserInfo Endpoint
        to encode the Claims in a JWT.
        """
        validate_array_value(self, "userinfo_encryption_enc_values_supported")

    def validate_request_object_signing_alg_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWS signing
        algorithms (alg values) supported by the OP for Request Objects,
        which are described in Section 6.1 of OpenID Connect Core 1.0.
        These algorithms are used both when the Request Object is passed
        by value (using the request parameter) and when it is passed by
        reference (using the request_uri parameter). Servers SHOULD support
        none and RS256.
        """
        values = self.get("request_object_signing_alg_values_supported")
        if not values:
            return

        if not isinstance(values, list):
            raise ValueError(
                '"request_object_signing_alg_values_supported" MUST be JSON array'
            )

    def validate_request_object_encryption_alg_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWE encryption
        algorithms (alg values) supported by the OP for Request Objects.
        These algorithms are used both when the Request Object is passed
        by value and when it is passed by reference.
        """
        validate_array_value(self, "request_object_encryption_alg_values_supported")

    def validate_request_object_encryption_enc_values_supported(self):
        """OPTIONAL. JSON array containing a list of the JWE encryption
        algorithms (enc values) supported by the OP for Request Objects.
        These algorithms are used both when the Request Object is passed
        by value and when it is passed by reference.
        """
        validate_array_value(self, "request_object_encryption_enc_values_supported")

    def validate_display_values_supported(self):
        """OPTIONAL. JSON array containing a list of the display parameter
        values that the OpenID Provider supports. These values are described
        in Section 3.1.2.1 of OpenID Connect Core 1.0.
        """
        values = self.get("display_values_supported")
        if not values:
            return

        if not isinstance(values, list):
            raise ValueError('"display_values_supported" MUST be JSON array')

        valid_values = {"page", "popup", "touch", "wap"}
        if not valid_values.issuperset(set(values)):
            raise ValueError('"display_values_supported" contains invalid values')

    def validate_claim_types_supported(self):
        """OPTIONAL. JSON array containing a list of the Claim Types that
        the OpenID Provider supports. These Claim Types are described in
        Section 5.6 of OpenID Connect Core 1.0. Values defined by this
        specification are normal, aggregated, and distributed. If omitted,
        the implementation supports only normal Claims.
        """
        values = self.get("claim_types_supported")
        if not values:
            return

        if not isinstance(values, list):
            raise ValueError('"claim_types_supported" MUST be JSON array')

        valid_values = {"normal", "aggregated", "distributed"}
        if not valid_values.issuperset(set(values)):
            raise ValueError('"claim_types_supported" contains invalid values')

    def validate_claims_supported(self):
        """RECOMMENDED. JSON array containing a list of the Claim Names
        of the Claims that the OpenID Provider MAY be able to supply values
        for. Note that for privacy or other reasons, this might not be an
        exhaustive list.
        """
        validate_array_value(self, "claims_supported")

    def validate_claims_locales_supported(self):
        """OPTIONAL. Languages and scripts supported for values in Claims
        being returned, represented as a JSON array of BCP47 [RFC5646]
        language tag values. Not all languages and scripts are necessarily
        supported for all Claim values.
        """
        validate_array_value(self, "claims_locales_supported")

    def validate_claims_parameter_supported(self):
        """OPTIONAL. Boolean value specifying whether the OP supports use of
        the claims parameter, with true indicating support. If omitted, the
        default value is false.
        """
        _validate_boolean_value(self, "claims_parameter_supported")

    def validate_request_parameter_supported(self):
        """OPTIONAL. Boolean value specifying whether the OP supports use of
        the request parameter, with true indicating support. If omitted, the
        default value is false.
        """
        _validate_boolean_value(self, "request_parameter_supported")

    def validate_request_uri_parameter_supported(self):
        """OPTIONAL. Boolean value specifying whether the OP supports use of
        the request_uri parameter, with true indicating support. If omitted,
        the default value is true.
        """
        _validate_boolean_value(self, "request_uri_parameter_supported")

    def validate_require_request_uri_registration(self):
        """OPTIONAL. Boolean value specifying whether the OP requires any
        request_uri values used to be pre-registered using the request_uris
        registration parameter. Pre-registration is REQUIRED when the value
        is true. If omitted, the default value is false.
        """
        _validate_boolean_value(self, "require_request_uri_registration")

    @property
    def claim_types_supported(self):
        # If omitted, the implementation supports only normal Claims
        return self.get("claim_types_supported", ["normal"])

    @property
    def claims_parameter_supported(self):
        # If omitted, the default value is false.
        return self.get("claims_parameter_supported", False)

    @property
    def request_parameter_supported(self):
        # If omitted, the default value is false.
        return self.get("request_parameter_supported", False)

    @property
    def request_uri_parameter_supported(self):
        # If omitted, the default value is true.
        return self.get("request_uri_parameter_supported", True)

    @property
    def require_request_uri_registration(self):
        # If omitted, the default value is false.
        return self.get("require_request_uri_registration", False)


def _validate_boolean_value(metadata, key):
    if key not in metadata:
        return
    if metadata[key] not in (True, False):
        raise ValueError(f'"{key}" MUST be boolean')
