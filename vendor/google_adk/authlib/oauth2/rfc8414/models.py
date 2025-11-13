from authlib.common.security import is_secure_transport
from authlib.common.urls import is_valid_url
from authlib.common.urls import urlparse


class AuthorizationServerMetadata(dict):
    """Define Authorization Server Metadata via `Section 2`_ in RFC8414_.

    .. _RFC8414: https://tools.ietf.org/html/rfc8414
    .. _`Section 2`: https://tools.ietf.org/html/rfc8414#section-2
    """

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
        "token_endpoint_auth_signing_alg_values_supported",
        "service_documentation",
        "ui_locales_supported",
        "op_policy_uri",
        "op_tos_uri",
        "revocation_endpoint",
        "revocation_endpoint_auth_methods_supported",
        "revocation_endpoint_auth_signing_alg_values_supported",
        "introspection_endpoint",
        "introspection_endpoint_auth_methods_supported",
        "introspection_endpoint_auth_signing_alg_values_supported",
        "code_challenge_methods_supported",
    ]

    def validate_issuer(self):
        """REQUIRED. The authorization server's issuer identifier, which is
        a URL that uses the "https" scheme and has no query or fragment
        components.
        """
        issuer = self.get("issuer")

        #: 1. REQUIRED
        if not issuer:
            raise ValueError('"issuer" is required')

        parsed = urlparse.urlparse(issuer)

        #: 2. uses the "https" scheme
        if not is_secure_transport(issuer):
            raise ValueError('"issuer" MUST use "https" scheme')

        #: 3. has no query or fragment
        if parsed.query or parsed.fragment:
            raise ValueError('"issuer" has no query or fragment')

    def validate_authorization_endpoint(self):
        """URL of the authorization server's authorization endpoint
        [RFC6749]. This is REQUIRED unless no grant types are supported
        that use the authorization endpoint.
        """
        url = self.get("authorization_endpoint")
        if url:
            if not is_secure_transport(url):
                raise ValueError('"authorization_endpoint" MUST use "https" scheme')
            return

        grant_types_supported = set(self.grant_types_supported)
        authorization_grant_types = {"authorization_code", "implicit"}
        if grant_types_supported & authorization_grant_types:
            raise ValueError('"authorization_endpoint" is required')

    def validate_token_endpoint(self):
        """URL of the authorization server's token endpoint [RFC6749]. This
        is REQUIRED unless only the implicit grant type is supported.
        """
        grant_types_supported = self.get("grant_types_supported")
        if (
            grant_types_supported
            and len(grant_types_supported) == 1
            and grant_types_supported[0] == "implicit"
        ):
            return

        url = self.get("token_endpoint")
        if not url:
            raise ValueError('"token_endpoint" is required')

        if not is_secure_transport(url):
            raise ValueError('"token_endpoint" MUST use "https" scheme')

    def validate_jwks_uri(self):
        """OPTIONAL.  URL of the authorization server's JWK Set [JWK]
        document.  The referenced document contains the signing key(s) the
        client uses to validate signatures from the authorization server.
        This URL MUST use the "https" scheme.  The JWK Set MAY also
        contain the server's encryption key or keys, which are used by
        clients to encrypt requests to the server.  When both signing and
        encryption keys are made available, a "use" (public key use)
        parameter value is REQUIRED for all keys in the referenced JWK Set
        to indicate each key's intended usage.
        """
        url = self.get("jwks_uri")
        if url and not is_secure_transport(url):
            raise ValueError('"jwks_uri" MUST use "https" scheme')

    def validate_registration_endpoint(self):
        """OPTIONAL.  URL of the authorization server's OAuth 2.0 Dynamic
        Client Registration endpoint [RFC7591].
        """
        url = self.get("registration_endpoint")
        if url and not is_secure_transport(url):
            raise ValueError('"registration_endpoint" MUST use "https" scheme')

    def validate_scopes_supported(self):
        """RECOMMENDED. JSON array containing a list of the OAuth 2.0
        [RFC6749] "scope" values that this authorization server supports.
        Servers MAY choose not to advertise some supported scope values
        even when this parameter is used.
        """
        validate_array_value(self, "scopes_supported")

    def validate_response_types_supported(self):
        """REQUIRED.  JSON array containing a list of the OAuth 2.0
        "response_type" values that this authorization server supports.
        The array values used are the same as those used with the
        "response_types" parameter defined by "OAuth 2.0 Dynamic Client
        Registration Protocol" [RFC7591].
        """
        response_types_supported = self.get("response_types_supported")
        if not response_types_supported:
            raise ValueError('"response_types_supported" is required')
        if not isinstance(response_types_supported, list):
            raise ValueError('"response_types_supported" MUST be JSON array')

    def validate_response_modes_supported(self):
        """OPTIONAL.  JSON array containing a list of the OAuth 2.0
        "response_mode" values that this authorization server supports, as
        specified in "OAuth 2.0 Multiple Response Type Encoding Practices"
        [OAuth.Responses].  If omitted, the default is "["query",
        "fragment"]".  The response mode value "form_post" is also defined
        in "OAuth 2.0 Form Post Response Mode" [OAuth.Post].
        """
        validate_array_value(self, "response_modes_supported")

    def validate_grant_types_supported(self):
        """OPTIONAL. JSON array containing a list of the OAuth 2.0 grant
        type values that this authorization server supports.  The array
        values used are the same as those used with the "grant_types"
        parameter defined by "OAuth 2.0 Dynamic Client Registration
        Protocol" [RFC7591].  If omitted, the default value is
        "["authorization_code", "implicit"]".
        """
        validate_array_value(self, "grant_types_supported")

    def validate_token_endpoint_auth_methods_supported(self):
        """OPTIONAL.  JSON array containing a list of client authentication
        methods supported by this token endpoint.  Client authentication
        method values are used in the "token_endpoint_auth_method"
        parameter defined in Section 2 of [RFC7591].  If omitted, the
        default is "client_secret_basic" -- the HTTP Basic Authentication
        Scheme specified in Section 2.3.1 of OAuth 2.0 [RFC6749].
        """
        validate_array_value(self, "token_endpoint_auth_methods_supported")

    def validate_token_endpoint_auth_signing_alg_values_supported(self):
        """OPTIONAL.  JSON array containing a list of the JWS signing
        algorithms ("alg" values) supported by the token endpoint for the
        signature on the JWT [JWT] used to authenticate the client at the
        token endpoint for the "private_key_jwt" and "client_secret_jwt"
        authentication methods.  This metadata entry MUST be present if
        either of these authentication methods are specified in the
        "token_endpoint_auth_methods_supported" entry.  No default
        algorithms are implied if this entry is omitted.  Servers SHOULD
        support "RS256".  The value "none" MUST NOT be used.
        """
        _validate_alg_values(
            self,
            "token_endpoint_auth_signing_alg_values_supported",
            self.token_endpoint_auth_methods_supported,
        )

    def validate_service_documentation(self):
        """OPTIONAL. URL of a page containing human-readable information
        that developers might want or need to know when using the
        authorization server.  In particular, if the authorization server
        does not support Dynamic Client Registration, then information on
        how to register clients needs to be provided in this
        documentation.
        """
        value = self.get("service_documentation")
        if value and not is_valid_url(value):
            raise ValueError('"service_documentation" MUST be a URL')

    def validate_ui_locales_supported(self):
        """OPTIONAL.  Languages and scripts supported for the user interface,
        represented as a JSON array of language tag values from BCP 47
        [RFC5646].  If omitted, the set of supported languages and scripts
        is unspecified.
        """
        validate_array_value(self, "ui_locales_supported")

    def validate_op_policy_uri(self):
        """OPTIONAL.  URL that the authorization server provides to the
        person registering the client to read about the authorization
        server's requirements on how the client can use the data provided
        by the authorization server.  The registration process SHOULD
        display this URL to the person registering the client if it is
        given.  As described in Section 5, despite the identifier
        "op_policy_uri" appearing to be OpenID-specific, its usage in this
        specification is actually referring to a general OAuth 2.0 feature
        that is not specific to OpenID Connect.
        """
        value = self.get("op_policy_uri")
        if value and not is_valid_url(value):
            raise ValueError('"op_policy_uri" MUST be a URL')

    def validate_op_tos_uri(self):
        """OPTIONAL.  URL that the authorization server provides to the
        person registering the client to read about the authorization
        server's terms of service.  The registration process SHOULD
        display this URL to the person registering the client if it is
        given.  As described in Section 5, despite the identifier
        "op_tos_uri", appearing to be OpenID-specific, its usage in this
        specification is actually referring to a general OAuth 2.0 feature
        that is not specific to OpenID Connect.
        """
        value = self.get("op_tos_uri")
        if value and not is_valid_url(value):
            raise ValueError('"op_tos_uri" MUST be a URL')

    def validate_revocation_endpoint(self):
        """OPTIONAL. URL of the authorization server's OAuth 2.0 revocation
        endpoint [RFC7009].
        """
        url = self.get("revocation_endpoint")
        if url and not is_secure_transport(url):
            raise ValueError('"revocation_endpoint" MUST use "https" scheme')

    def validate_revocation_endpoint_auth_methods_supported(self):
        """OPTIONAL.  JSON array containing a list of client authentication
        methods supported by this revocation endpoint.  The valid client
        authentication method values are those registered in the IANA
        "OAuth Token Endpoint Authentication Methods" registry
        [IANA.OAuth.Parameters].  If omitted, the default is
        "client_secret_basic" -- the HTTP Basic Authentication Scheme
        specified in Section 2.3.1 of OAuth 2.0 [RFC6749].
        """
        validate_array_value(self, "revocation_endpoint_auth_methods_supported")

    def validate_revocation_endpoint_auth_signing_alg_values_supported(self):
        """OPTIONAL.  JSON array containing a list of the JWS signing
        algorithms ("alg" values) supported by the revocation endpoint for
        the signature on the JWT [JWT] used to authenticate the client at
        the revocation endpoint for the "private_key_jwt" and
        "client_secret_jwt" authentication methods.  This metadata entry
        MUST be present if either of these authentication methods are
        specified in the "revocation_endpoint_auth_methods_supported"
        entry.  No default algorithms are implied if this entry is
        omitted.  The value "none" MUST NOT be used.
        """
        _validate_alg_values(
            self,
            "revocation_endpoint_auth_signing_alg_values_supported",
            self.revocation_endpoint_auth_methods_supported,
        )

    def validate_introspection_endpoint(self):
        """OPTIONAL.  URL of the authorization server's OAuth 2.0
        introspection endpoint [RFC7662].
        """
        url = self.get("introspection_endpoint")
        if url and not is_secure_transport(url):
            raise ValueError('"introspection_endpoint" MUST use "https" scheme')

    def validate_introspection_endpoint_auth_methods_supported(self):
        """OPTIONAL.  JSON array containing a list of client authentication
        methods supported by this introspection endpoint.  The valid
        client authentication method values are those registered in the
        IANA "OAuth Token Endpoint Authentication Methods" registry
        [IANA.OAuth.Parameters] or those registered in the IANA "OAuth
        Access Token Types" registry [IANA.OAuth.Parameters].  (These
        values are and will remain distinct, due to Section 7.2.)  If
        omitted, the set of supported authentication methods MUST be
        determined by other means.
        """
        validate_array_value(self, "introspection_endpoint_auth_methods_supported")

    def validate_introspection_endpoint_auth_signing_alg_values_supported(self):
        """OPTIONAL.  JSON array containing a list of the JWS signing
        algorithms ("alg" values) supported by the introspection endpoint
        for the signature on the JWT [JWT] used to authenticate the client
        at the introspection endpoint for the "private_key_jwt" and
        "client_secret_jwt" authentication methods.  This metadata entry
        MUST be present if either of these authentication methods are
        specified in the "introspection_endpoint_auth_methods_supported"
        entry.  No default algorithms are implied if this entry is
        omitted.  The value "none" MUST NOT be used.
        """
        _validate_alg_values(
            self,
            "introspection_endpoint_auth_signing_alg_values_supported",
            self.introspection_endpoint_auth_methods_supported,
        )

    def validate_code_challenge_methods_supported(self):
        """OPTIONAL.  JSON array containing a list of Proof Key for Code
        Exchange (PKCE) [RFC7636] code challenge methods supported by this
        authorization server.  Code challenge method values are used in
        the "code_challenge_method" parameter defined in Section 4.3 of
        [RFC7636].  The valid code challenge method values are those
        registered in the IANA "PKCE Code Challenge Methods" registry
        [IANA.OAuth.Parameters].  If omitted, the authorization server
        does not support PKCE.
        """
        validate_array_value(self, "code_challenge_methods_supported")

    @property
    def response_modes_supported(self):
        #: If omitted, the default is ["query", "fragment"]
        return self.get("response_modes_supported", ["query", "fragment"])

    @property
    def grant_types_supported(self):
        #: If omitted, the default value is ["authorization_code", "implicit"]
        return self.get("grant_types_supported", ["authorization_code", "implicit"])

    @property
    def token_endpoint_auth_methods_supported(self):
        #: If omitted, the default is "client_secret_basic"
        return self.get(
            "token_endpoint_auth_methods_supported", ["client_secret_basic"]
        )

    @property
    def revocation_endpoint_auth_methods_supported(self):
        #: If omitted, the default is "client_secret_basic"
        return self.get(
            "revocation_endpoint_auth_methods_supported", ["client_secret_basic"]
        )

    @property
    def introspection_endpoint_auth_methods_supported(self):
        #: If omitted, the set of supported authentication methods MUST be
        #: determined by other means
        #: here, we use "client_secret_basic"
        return self.get(
            "introspection_endpoint_auth_methods_supported", ["client_secret_basic"]
        )

    def validate(self):
        """Validate all server metadata value."""
        for key in self.REGISTRY_KEYS:
            object.__getattribute__(self, f"validate_{key}")()

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as error:
            if key in self.REGISTRY_KEYS:
                return self.get(key)
            raise error


def _validate_alg_values(data, key, auth_methods_supported):
    value = data.get(key)
    if value and not isinstance(value, list):
        raise ValueError(f'"{key}" MUST be JSON array')

    auth_methods = set(auth_methods_supported)
    jwt_auth_methods = {"private_key_jwt", "client_secret_jwt"}
    if auth_methods & jwt_auth_methods:
        if not value:
            raise ValueError(f'"{key}" is required')

    if value and "none" in value:
        raise ValueError(f'the value "none" MUST NOT be used in "{key}"')


def validate_array_value(metadata, key):
    values = metadata.get(key)
    if values is not None and not isinstance(values, list):
        raise ValueError(f'"{key}" MUST be JSON array')
