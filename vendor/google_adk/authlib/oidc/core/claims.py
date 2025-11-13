import hmac
import time

from authlib.common.encoding import to_bytes
from authlib.jose import JWTClaims
from authlib.jose.errors import InvalidClaimError
from authlib.jose.errors import MissingClaimError
from authlib.oauth2.rfc6749.util import scope_to_list

from .util import create_half_hash

__all__ = [
    "IDToken",
    "CodeIDToken",
    "ImplicitIDToken",
    "HybridIDToken",
    "UserInfo",
    "get_claim_cls_by_response_type",
]

_REGISTERED_CLAIMS = [
    "iss",
    "sub",
    "aud",
    "exp",
    "nbf",
    "iat",
    "auth_time",
    "nonce",
    "acr",
    "amr",
    "azp",
    "at_hash",
]


class IDToken(JWTClaims):
    ESSENTIAL_CLAIMS = ["iss", "sub", "aud", "exp", "iat"]

    def validate(self, now=None, leeway=0):
        for k in self.ESSENTIAL_CLAIMS:
            if k not in self:
                raise MissingClaimError(k)

        self._validate_essential_claims()
        if now is None:
            now = int(time.time())

        self.validate_iss()
        self.validate_sub()
        self.validate_aud()
        self.validate_exp(now, leeway)
        self.validate_nbf(now, leeway)
        self.validate_iat(now, leeway)
        self.validate_auth_time()
        self.validate_nonce()
        self.validate_acr()
        self.validate_amr()
        self.validate_azp()
        self.validate_at_hash()

    def validate_auth_time(self):
        """Time when the End-User authentication occurred. Its value is a JSON
        number representing the number of seconds from 1970-01-01T0:0:0Z as
        measured in UTC until the date/time. When a max_age request is made or
        when auth_time is requested as an Essential Claim, then this Claim is
        REQUIRED; otherwise, its inclusion is OPTIONAL.
        """
        auth_time = self.get("auth_time")
        if self.params.get("max_age") and not auth_time:
            raise MissingClaimError("auth_time")

        if auth_time and not isinstance(auth_time, (int, float)):
            raise InvalidClaimError("auth_time")

    def validate_nonce(self):
        """String value used to associate a Client session with an ID Token,
        and to mitigate replay attacks. The value is passed through unmodified
        from the Authentication Request to the ID Token. If present in the ID
        Token, Clients MUST verify that the nonce Claim Value is equal to the
        value of the nonce parameter sent in the Authentication Request. If
        present in the Authentication Request, Authorization Servers MUST
        include a nonce Claim in the ID Token with the Claim Value being the
        nonce value sent in the Authentication Request. Authorization Servers
        SHOULD perform no other processing on nonce values used. The nonce
        value is a case sensitive string.
        """
        nonce_value = self.params.get("nonce")
        if nonce_value:
            if "nonce" not in self:
                raise MissingClaimError("nonce")
            if nonce_value != self["nonce"]:
                raise InvalidClaimError("nonce")

    def validate_acr(self):
        """OPTIONAL. Authentication Context Class Reference. String specifying
        an Authentication Context Class Reference value that identifies the
        Authentication Context Class that the authentication performed
        satisfied. The value "0" indicates the End-User authentication did not
        meet the requirements of `ISO/IEC 29115`_ level 1. Authentication
        using a long-lived browser cookie, for instance, is one example where
        the use of "level 0" is appropriate. Authentications with level 0
        SHOULD NOT be used to authorize access to any resource of any monetary
        value. An absolute URI or an `RFC 6711`_ registered name SHOULD be
        used as the acr value; registered names MUST NOT be used with a
        different meaning than that which is registered. Parties using this
        claim will need to agree upon the meanings of the values used, which
        may be context-specific. The acr value is a case sensitive string.

        .. _`ISO/IEC 29115`: https://www.iso.org/standard/45138.html
        .. _`RFC 6711`: https://tools.ietf.org/html/rfc6711
        """
        return self._validate_claim_value("acr")

    def validate_amr(self):
        """OPTIONAL. Authentication Methods References. JSON array of strings
        that are identifiers for authentication methods used in the
        authentication. For instance, values might indicate that both password
        and OTP authentication methods were used. The definition of particular
        values to be used in the amr Claim is beyond the scope of this
        specification. Parties using this claim will need to agree upon the
        meanings of the values used, which may be context-specific. The amr
        value is an array of case sensitive strings.
        """
        amr = self.get("amr")
        if amr and not isinstance(self["amr"], list):
            raise InvalidClaimError("amr")

    def validate_azp(self):
        """OPTIONAL. Authorized party - the party to which the ID Token was
        issued. If present, it MUST contain the OAuth 2.0 Client ID of this
        party. This Claim is only needed when the ID Token has a single
        audience value and that audience is different than the authorized
        party. It MAY be included even when the authorized party is the same
        as the sole audience. The azp value is a case sensitive string
        containing a StringOrURI value.
        """
        aud = self.get("aud")
        client_id = self.params.get("client_id")
        required = False
        if aud and client_id:
            if isinstance(aud, list) and len(aud) == 1:
                aud = aud[0]
            if aud != client_id:
                required = True

        azp = self.get("azp")
        if required and not azp:
            raise MissingClaimError("azp")

        if azp and client_id and azp != client_id:
            raise InvalidClaimError("azp")

    def validate_at_hash(self):
        """OPTIONAL. Access Token hash value. Its value is the base64url
        encoding of the left-most half of the hash of the octets of the ASCII
        representation of the access_token value, where the hash algorithm
        used is the hash algorithm used in the alg Header Parameter of the
        ID Token's JOSE Header. For instance, if the alg is RS256, hash the
        access_token value with SHA-256, then take the left-most 128 bits and
        base64url encode them. The at_hash value is a case sensitive string.
        """
        access_token = self.params.get("access_token")
        at_hash = self.get("at_hash")
        if at_hash and access_token:
            if not _verify_hash(at_hash, access_token, self.header["alg"]):
                raise InvalidClaimError("at_hash")


class CodeIDToken(IDToken):
    RESPONSE_TYPES = ("code",)
    REGISTERED_CLAIMS = _REGISTERED_CLAIMS


class ImplicitIDToken(IDToken):
    RESPONSE_TYPES = ("id_token", "id_token token")
    ESSENTIAL_CLAIMS = ["iss", "sub", "aud", "exp", "iat", "nonce"]
    REGISTERED_CLAIMS = _REGISTERED_CLAIMS

    def validate_at_hash(self):
        """If the ID Token is issued from the Authorization Endpoint with an
        access_token value, which is the case for the response_type value
        id_token token, this is REQUIRED; it MAY NOT be used when no Access
        Token is issued, which is the case for the response_type value
        id_token.
        """
        access_token = self.params.get("access_token")
        if access_token and "at_hash" not in self:
            raise MissingClaimError("at_hash")
        super().validate_at_hash()


class HybridIDToken(ImplicitIDToken):
    RESPONSE_TYPES = ("code id_token", "code token", "code id_token token")
    REGISTERED_CLAIMS = _REGISTERED_CLAIMS + ["c_hash"]

    def validate(self, now=None, leeway=0):
        super().validate(now=now, leeway=leeway)
        self.validate_c_hash()

    def validate_c_hash(self):
        """Code hash value. Its value is the base64url encoding of the
        left-most half of the hash of the octets of the ASCII representation
        of the code value, where the hash algorithm used is the hash algorithm
        used in the alg Header Parameter of the ID Token's JOSE Header. For
        instance, if the alg is HS512, hash the code value with SHA-512, then
        take the left-most 256 bits and base64url encode them. The c_hash
        value is a case sensitive string.
        If the ID Token is issued from the Authorization Endpoint with a code,
        which is the case for the response_type values code id_token and code
        id_token token, this is REQUIRED; otherwise, its inclusion is OPTIONAL.
        """
        code = self.params.get("code")
        c_hash = self.get("c_hash")
        if code:
            if not c_hash:
                raise MissingClaimError("c_hash")
            if not _verify_hash(c_hash, code, self.header["alg"]):
                raise InvalidClaimError("c_hash")


class UserInfo(dict):
    """The standard claims of a UserInfo object. Defined per `Section 5.1`_.

    .. _`Section 5.1`: http://openid.net/specs/openid-connect-core-1_0.html#StandardClaims
    """

    #: registered claims that UserInfo supports
    REGISTERED_CLAIMS = [
        "sub",
        "name",
        "given_name",
        "family_name",
        "middle_name",
        "nickname",
        "preferred_username",
        "profile",
        "picture",
        "website",
        "email",
        "email_verified",
        "gender",
        "birthdate",
        "zoneinfo",
        "locale",
        "phone_number",
        "phone_number_verified",
        "address",
        "updated_at",
    ]

    SCOPES_CLAIMS_MAPPING = {
        "openid": ["sub"],
        "profile": [
            "name",
            "family_name",
            "given_name",
            "middle_name",
            "nickname",
            "preferred_username",
            "profile",
            "picture",
            "website",
            "gender",
            "birthdate",
            "zoneinfo",
            "locale",
            "updated_at",
        ],
        "email": ["email", "email_verified"],
        "address": ["address"],
        "phone": ["phone_number", "phone_number_verified"],
    }

    def filter(self, scope: str):
        """Return a new UserInfo object containing only the claims matching the scope passed in parameter."""
        scope = scope_to_list(scope)
        filtered_claims = [
            claim
            for scope_part in scope
            for claim in self.SCOPES_CLAIMS_MAPPING.get(scope_part, [])
        ]
        filtered_items = {
            key: val for key, val in self.items() if key in filtered_claims
        }
        return UserInfo(filtered_items)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as error:
            if key in self.REGISTERED_CLAIMS:
                return self.get(key)
            raise error


def get_claim_cls_by_response_type(response_type):
    claims_classes = (CodeIDToken, ImplicitIDToken, HybridIDToken)
    for claims_cls in claims_classes:
        if response_type in claims_cls.RESPONSE_TYPES:
            return claims_cls


def _verify_hash(signature, s, alg):
    hash_value = create_half_hash(s, alg)
    if not hash_value:
        return True
    return hmac.compare_digest(hash_value, to_bytes(signature))
