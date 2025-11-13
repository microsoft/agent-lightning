from authlib.jose.errors import InvalidClaimError
from authlib.jose.rfc7519 import JWTClaims


class JWTAccessTokenClaims(JWTClaims):
    REGISTERED_CLAIMS = JWTClaims.REGISTERED_CLAIMS + [
        "client_id",
        "auth_time",
        "acr",
        "amr",
        "scope",
        "groups",
        "roles",
        "entitlements",
    ]

    def validate(self, **kwargs):
        self.validate_typ()

        super().validate(**kwargs)
        self.validate_client_id()
        self.validate_auth_time()
        self.validate_acr()
        self.validate_amr()
        self.validate_scope()
        self.validate_groups()
        self.validate_roles()
        self.validate_entitlements()

    def validate_typ(self):
        # The resource server MUST verify that the 'typ' header value is 'at+jwt'
        # or 'application/at+jwt' and reject tokens carrying any other value.
        # 'typ' is not a required claim, so we don't raise an error if it's missing.
        typ = self.header.get("typ")
        if typ and typ.lower() not in ("at+jwt", "application/at+jwt"):
            raise InvalidClaimError("typ")

    def validate_client_id(self):
        return self._validate_claim_value("client_id")

    def validate_auth_time(self):
        auth_time = self.get("auth_time")
        if auth_time and not isinstance(auth_time, (int, float)):
            raise InvalidClaimError("auth_time")

    def validate_acr(self):
        return self._validate_claim_value("acr")

    def validate_amr(self):
        amr = self.get("amr")
        if amr and not isinstance(self["amr"], list):
            raise InvalidClaimError("amr")

    def validate_scope(self):
        return self._validate_claim_value("scope")

    def validate_groups(self):
        return self._validate_claim_value("groups")

    def validate_roles(self):
        return self._validate_claim_value("roles")

    def validate_entitlements(self):
        return self._validate_claim_value("entitlements")
