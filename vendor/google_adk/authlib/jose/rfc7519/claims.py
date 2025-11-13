import time

from authlib.jose.errors import ExpiredTokenError
from authlib.jose.errors import InvalidClaimError
from authlib.jose.errors import InvalidTokenError
from authlib.jose.errors import MissingClaimError


class BaseClaims(dict):
    """Payload claims for JWT, which contains a validate interface.

    :param payload: the payload dict of JWT
    :param header: the header dict of JWT
    :param options: validate options
    :param params: other params

    An example on ``options`` parameter, the format is inspired by
    `OpenID Connect Claims`_::

        {
            "iss": {
                "essential": True,
                "values": ["https://example.com", "https://example.org"]
            },
            "sub": {
                "essential": True
                "value": "248289761001"
            },
            "jti": {
                "validate": validate_jti
            }
        }

    .. _`OpenID Connect Claims`:
        http://openid.net/specs/openid-connect-core-1_0.html#IndividualClaimsRequests
    """

    REGISTERED_CLAIMS = []

    def __init__(self, payload, header, options=None, params=None):
        super().__init__(payload)
        self.header = header
        self.options = options or {}
        self.params = params or {}

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as error:
            if key in self.REGISTERED_CLAIMS:
                return self.get(key)
            raise error

    def _validate_essential_claims(self):
        for k in self.options:
            if self.options[k].get("essential"):
                if k not in self:
                    raise MissingClaimError(k)
                elif not self.get(k):
                    raise InvalidClaimError(k)

    def _validate_claim_value(self, claim_name):
        option = self.options.get(claim_name)
        if not option:
            return

        value = self.get(claim_name)
        option_value = option.get("value")
        if option_value and value != option_value:
            raise InvalidClaimError(claim_name)

        option_values = option.get("values")
        if option_values and value not in option_values:
            raise InvalidClaimError(claim_name)

        validate = option.get("validate")
        if validate and not validate(self, value):
            raise InvalidClaimError(claim_name)

    def get_registered_claims(self):
        rv = {}
        for k in self.REGISTERED_CLAIMS:
            if k in self:
                rv[k] = self[k]
        return rv


class JWTClaims(BaseClaims):
    REGISTERED_CLAIMS = ["iss", "sub", "aud", "exp", "nbf", "iat", "jti"]

    def validate(self, now=None, leeway=0):
        """Validate everything in claims payload."""
        self._validate_essential_claims()

        if now is None:
            now = int(time.time())

        self.validate_iss()
        self.validate_sub()
        self.validate_aud()
        self.validate_exp(now, leeway)
        self.validate_nbf(now, leeway)
        self.validate_iat(now, leeway)
        self.validate_jti()

        # Validate custom claims
        for key in self.options.keys():
            if key not in self.REGISTERED_CLAIMS:
                self._validate_claim_value(key)

    def validate_iss(self):
        """The "iss" (issuer) claim identifies the principal that issued the
        JWT.  The processing of this claim is generally application specific.
        The "iss" value is a case-sensitive string containing a StringOrURI
        value.  Use of this claim is OPTIONAL.
        """
        self._validate_claim_value("iss")

    def validate_sub(self):
        """The "sub" (subject) claim identifies the principal that is the
        subject of the JWT.  The claims in a JWT are normally statements
        about the subject.  The subject value MUST either be scoped to be
        locally unique in the context of the issuer or be globally unique.
        The processing of this claim is generally application specific.  The
        "sub" value is a case-sensitive string containing a StringOrURI
        value.  Use of this claim is OPTIONAL.
        """
        self._validate_claim_value("sub")

    def validate_aud(self):
        """The "aud" (audience) claim identifies the recipients that the JWT is
        intended for.  Each principal intended to process the JWT MUST
        identify itself with a value in the audience claim.  If the principal
        processing the claim does not identify itself with a value in the
        "aud" claim when this claim is present, then the JWT MUST be
        rejected.  In the general case, the "aud" value is an array of case-
        sensitive strings, each containing a StringOrURI value.  In the
        special case when the JWT has one audience, the "aud" value MAY be a
        single case-sensitive string containing a StringOrURI value.  The
        interpretation of audience values is generally application specific.
        Use of this claim is OPTIONAL.
        """
        aud_option = self.options.get("aud")
        aud = self.get("aud")
        if not aud_option or not aud:
            return

        aud_values = aud_option.get("values")
        if not aud_values:
            aud_value = aud_option.get("value")
            if aud_value:
                aud_values = [aud_value]

        if not aud_values:
            return

        if isinstance(self["aud"], list):
            aud_list = self["aud"]
        else:
            aud_list = [self["aud"]]

        if not any([v in aud_list for v in aud_values]):
            raise InvalidClaimError("aud")

    def validate_exp(self, now, leeway):
        """The "exp" (expiration time) claim identifies the expiration time on
        or after which the JWT MUST NOT be accepted for processing.  The
        processing of the "exp" claim requires that the current date/time
        MUST be before the expiration date/time listed in the "exp" claim.
        Implementers MAY provide for some small leeway, usually no more than
        a few minutes, to account for clock skew.  Its value MUST be a number
        containing a NumericDate value.  Use of this claim is OPTIONAL.
        """
        if "exp" in self:
            exp = self["exp"]
            if not _validate_numeric_time(exp):
                raise InvalidClaimError("exp")
            if exp < (now - leeway):
                raise ExpiredTokenError()

    def validate_nbf(self, now, leeway):
        """The "nbf" (not before) claim identifies the time before which the JWT
        MUST NOT be accepted for processing.  The processing of the "nbf"
        claim requires that the current date/time MUST be after or equal to
        the not-before date/time listed in the "nbf" claim.  Implementers MAY
        provide for some small leeway, usually no more than a few minutes, to
        account for clock skew.  Its value MUST be a number containing a
        NumericDate value.  Use of this claim is OPTIONAL.
        """
        if "nbf" in self:
            nbf = self["nbf"]
            if not _validate_numeric_time(nbf):
                raise InvalidClaimError("nbf")
            if nbf > (now + leeway):
                raise InvalidTokenError()

    def validate_iat(self, now, leeway):
        """The "iat" (issued at) claim identifies the time at which the JWT was
        issued.  This claim can be used to determine the age of the JWT.
        Implementers MAY provide for some small leeway, usually no more
        than a few minutes, to account for clock skew. Its value MUST be a
        number containing a NumericDate value.  Use of this claim is OPTIONAL.
        """
        if "iat" in self:
            iat = self["iat"]
            if not _validate_numeric_time(iat):
                raise InvalidClaimError("iat")
            if iat > (now + leeway):
                raise InvalidTokenError(
                    description="The token is not valid as it was issued in the future"
                )

    def validate_jti(self):
        """The "jti" (JWT ID) claim provides a unique identifier for the JWT.
        The identifier value MUST be assigned in a manner that ensures that
        there is a negligible probability that the same value will be
        accidentally assigned to a different data object; if the application
        uses multiple issuers, collisions MUST be prevented among values
        produced by different issuers as well.  The "jti" claim can be used
        to prevent the JWT from being replayed.  The "jti" value is a case-
        sensitive string.  Use of this claim is OPTIONAL.
        """
        self._validate_claim_value("jti")


def _validate_numeric_time(s):
    return isinstance(s, (int, float))
