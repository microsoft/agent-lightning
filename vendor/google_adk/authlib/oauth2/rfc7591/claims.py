from authlib.common.urls import is_valid_url
from authlib.jose import BaseClaims
from authlib.jose import JsonWebKey
from authlib.jose.errors import InvalidClaimError

from ..rfc6749 import scope_to_list


class ClientMetadataClaims(BaseClaims):
    # https://tools.ietf.org/html/rfc7591#section-2
    REGISTERED_CLAIMS = [
        "redirect_uris",
        "token_endpoint_auth_method",
        "grant_types",
        "response_types",
        "client_name",
        "client_uri",
        "logo_uri",
        "scope",
        "contacts",
        "tos_uri",
        "policy_uri",
        "jwks_uri",
        "jwks",
        "software_id",
        "software_version",
    ]

    def validate(self):
        self._validate_essential_claims()
        self.validate_redirect_uris()
        self.validate_token_endpoint_auth_method()
        self.validate_grant_types()
        self.validate_response_types()
        self.validate_client_name()
        self.validate_client_uri()
        self.validate_logo_uri()
        self.validate_scope()
        self.validate_contacts()
        self.validate_tos_uri()
        self.validate_policy_uri()
        self.validate_jwks_uri()
        self.validate_jwks()
        self.validate_software_id()
        self.validate_software_version()

    def validate_redirect_uris(self):
        """Array of redirection URI strings for use in redirect-based flows
        such as the authorization code and implicit flows.  As required by
        Section 2 of OAuth 2.0 [RFC6749], clients using flows with
        redirection MUST register their redirection URI values.
        Authorization servers that support dynamic registration for
        redirect-based flows MUST implement support for this metadata
        value.
        """
        uris = self.get("redirect_uris")
        if uris:
            for uri in uris:
                self._validate_uri("redirect_uris", uri)

    def validate_token_endpoint_auth_method(self):
        """String indicator of the requested authentication method for the
        token endpoint.
        """
        # If unspecified or omitted, the default is "client_secret_basic"
        if "token_endpoint_auth_method" not in self:
            self["token_endpoint_auth_method"] = "client_secret_basic"
        self._validate_claim_value("token_endpoint_auth_method")

    def validate_grant_types(self):
        """Array of OAuth 2.0 grant type strings that the client can use at
        the token endpoint.
        """
        self._validate_claim_value("grant_types")

    def validate_response_types(self):
        """Array of the OAuth 2.0 response type strings that the client can
        use at the authorization endpoint.
        """
        self._validate_claim_value("response_types")

    def validate_client_name(self):
        """Human-readable string name of the client to be presented to the
        end-user during authorization.  If omitted, the authorization
        server MAY display the raw "client_id" value to the end-user
        instead.  It is RECOMMENDED that clients always send this field.
        The value of this field MAY be internationalized, as described in
        Section 2.2.
        """

    def validate_client_uri(self):
        """URL string of a web page providing information about the client.
        If present, the server SHOULD display this URL to the end-user in
        a clickable fashion.  It is RECOMMENDED that clients always send
        this field.  The value of this field MUST point to a valid web
        page.  The value of this field MAY be internationalized, as
        described in Section 2.2.
        """
        self._validate_uri("client_uri")

    def validate_logo_uri(self):
        """URL string that references a logo for the client.  If present, the
        server SHOULD display this image to the end-user during approval.
        The value of this field MUST point to a valid image file.  The
        value of this field MAY be internationalized, as described in
        Section 2.2.
        """
        self._validate_uri("logo_uri")

    def validate_scope(self):
        """String containing a space-separated list of scope values (as
        described in Section 3.3 of OAuth 2.0 [RFC6749]) that the client
        can use when requesting access tokens.  The semantics of values in
        this list are service specific.  If omitted, an authorization
        server MAY register a client with a default set of scopes.
        """
        self._validate_claim_value("scope")

    def validate_contacts(self):
        """Array of strings representing ways to contact people responsible
        for this client, typically email addresses.  The authorization
        server MAY make these contact addresses available to end-users for
        support requests for the client.  See Section 6 for information on
        Privacy Considerations.
        """
        if "contacts" in self and not isinstance(self["contacts"], list):
            raise InvalidClaimError("contacts")

    def validate_tos_uri(self):
        """URL string that points to a human-readable terms of service
        document for the client that describes a contractual relationship
        between the end-user and the client that the end-user accepts when
        authorizing the client.  The authorization server SHOULD display
        this URL to the end-user if it is provided.  The value of this
        field MUST point to a valid web page.  The value of this field MAY
        be internationalized, as described in Section 2.2.
        """
        self._validate_uri("tos_uri")

    def validate_policy_uri(self):
        """URL string that points to a human-readable privacy policy document
        that describes how the deployment organization collects, uses,
        retains, and discloses personal data.  The authorization server
        SHOULD display this URL to the end-user if it is provided.  The
        value of this field MUST point to a valid web page.  The value of
        this field MAY be internationalized, as described in Section 2.2.
        """
        self._validate_uri("policy_uri")

    def validate_jwks_uri(self):
        """URL string referencing the client's JSON Web Key (JWK) Set
        [RFC7517] document, which contains the client's public keys.  The
        value of this field MUST point to a valid JWK Set document.  These
        keys can be used by higher-level protocols that use signing or
        encryption.  For instance, these keys might be used by some
        applications for validating signed requests made to the token
        endpoint when using JWTs for client authentication [RFC7523].  Use
        of this parameter is preferred over the "jwks" parameter, as it
        allows for easier key rotation.  The "jwks_uri" and "jwks"
        parameters MUST NOT both be present in the same request or
        response.
        """
        # TODO: use real HTTP library
        self._validate_uri("jwks_uri")

    def validate_jwks(self):
        """Client's JSON Web Key Set [RFC7517] document value, which contains
        the client's public keys.  The value of this field MUST be a JSON
        object containing a valid JWK Set.  These keys can be used by
        higher-level protocols that use signing or encryption.  This
        parameter is intended to be used by clients that cannot use the
        "jwks_uri" parameter, such as native clients that cannot host
        public URLs.  The "jwks_uri" and "jwks" parameters MUST NOT both
        be present in the same request or response.
        """
        if "jwks" in self:
            if "jwks_uri" in self:
                #  The "jwks_uri" and "jwks" parameters MUST NOT both  be present
                raise InvalidClaimError("jwks")

            jwks = self["jwks"]
            try:
                key_set = JsonWebKey.import_key_set(jwks)
                if not key_set:
                    raise InvalidClaimError("jwks")
            except ValueError as exc:
                raise InvalidClaimError("jwks") from exc

    def validate_software_id(self):
        """A unique identifier string (e.g., a Universally Unique Identifier
        (UUID)) assigned by the client developer or software publisher
        used by registration endpoints to identify the client software to
        be dynamically registered.  Unlike "client_id", which is issued by
        the authorization server and SHOULD vary between instances, the
        "software_id" SHOULD remain the same for all instances of the
        client software.  The "software_id" SHOULD remain the same across
        multiple updates or versions of the same piece of software.  The
        value of this field is not intended to be human readable and is
        usually opaque to the client and authorization server.
        """

    def validate_software_version(self):
        """A version identifier string for the client software identified by
        "software_id".  The value of the "software_version" SHOULD change
        on any update to the client software identified by the same
        "software_id".  The value of this field is intended to be compared
        using string equality matching and no other comparison semantics
        are defined by this specification.  The value of this field is
        outside the scope of this specification, but it is not intended to
        be human readable and is usually opaque to the client and
        authorization server.  The definition of what constitutes an
        update to client software that would trigger a change to this
        value is specific to the software itself and is outside the scope
        of this specification.
        """

    def _validate_uri(self, key, uri=None):
        if uri is None:
            uri = self.get(key)
        if uri and not is_valid_url(uri, fragments_allowed=False):
            raise InvalidClaimError(key)

    @classmethod
    def get_claims_options(cls, metadata):
        """Generate claims options validation from Authorization Server metadata."""
        scopes_supported = metadata.get("scopes_supported")
        response_types_supported = metadata.get("response_types_supported")
        grant_types_supported = metadata.get("grant_types_supported")
        auth_methods_supported = metadata.get("token_endpoint_auth_methods_supported")
        options = {}
        if scopes_supported is not None:
            scopes_supported = set(scopes_supported)

            def _validate_scope(claims, value):
                if not value:
                    return True
                scopes = set(scope_to_list(value))
                return scopes_supported.issuperset(scopes)

            options["scope"] = {"validate": _validate_scope}

        if response_types_supported is not None:
            response_types_supported = [
                set(items.split()) for items in response_types_supported
            ]

            def _validate_response_types(claims, value):
                # If omitted, the default is that the client will use only the "code"
                # response type.
                response_types = (
                    [set(items.split()) for items in value] if value else [{"code"}]
                )
                return all(
                    response_type in response_types_supported
                    for response_type in response_types
                )

            options["response_types"] = {"validate": _validate_response_types}

        if grant_types_supported is not None:
            grant_types_supported = set(grant_types_supported)

            def _validate_grant_types(claims, value):
                # If omitted, the default behavior is that the client will use only
                # the "authorization_code" Grant Type.
                grant_types = set(value) if value else {"authorization_code"}
                return grant_types_supported.issuperset(grant_types)

            options["grant_types"] = {"validate": _validate_grant_types}

        if auth_methods_supported is not None:
            options["token_endpoint_auth_method"] = {"values": auth_methods_supported}

        return options
