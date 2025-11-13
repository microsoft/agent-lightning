from authlib.common.errors import AuthlibBaseError


class JoseError(AuthlibBaseError):
    pass


class DecodeError(JoseError):
    error = "decode_error"


class MissingAlgorithmError(JoseError):
    error = "missing_algorithm"


class UnsupportedAlgorithmError(JoseError):
    error = "unsupported_algorithm"


class BadSignatureError(JoseError):
    error = "bad_signature"

    def __init__(self, result):
        super().__init__()
        self.result = result


class InvalidHeaderParameterNameError(JoseError):
    error = "invalid_header_parameter_name"

    def __init__(self, name):
        description = f"Invalid Header Parameter Name: {name}"
        super().__init__(description=description)


class InvalidCritHeaderParameterNameError(JoseError):
    error = "invalid_crit_header_parameter_name"

    def __init__(self, name):
        description = f"Invalid Header Parameter Name: {name}"
        super().__init__(description=description)


class InvalidEncryptionAlgorithmForECDH1PUWithKeyWrappingError(JoseError):
    error = "invalid_encryption_algorithm_for_ECDH_1PU_with_key_wrapping"

    def __init__(self):
        description = (
            "In key agreement with key wrapping mode ECDH-1PU algorithm "
            "only supports AES_CBC_HMAC_SHA2 family encryption algorithms"
        )
        super().__init__(description=description)


class InvalidAlgorithmForMultipleRecipientsMode(JoseError):
    error = "invalid_algorithm_for_multiple_recipients_mode"

    def __init__(self, alg):
        description = f"{alg} algorithm cannot be used in multiple recipients mode"
        super().__init__(description=description)


class KeyMismatchError(JoseError):
    error = "key_mismatch_error"
    description = "Key does not match to any recipient"


class MissingEncryptionAlgorithmError(JoseError):
    error = "missing_encryption_algorithm"
    description = "Missing 'enc' in header"


class UnsupportedEncryptionAlgorithmError(JoseError):
    error = "unsupported_encryption_algorithm"
    description = "Unsupported 'enc' value in header"


class UnsupportedCompressionAlgorithmError(JoseError):
    error = "unsupported_compression_algorithm"
    description = "Unsupported 'zip' value in header"


class InvalidUseError(JoseError):
    error = "invalid_use"
    description = "Key 'use' is not valid for your usage"


class InvalidClaimError(JoseError):
    error = "invalid_claim"

    def __init__(self, claim):
        self.claim_name = claim
        description = f"Invalid claim '{claim}'"
        super().__init__(description=description)


class MissingClaimError(JoseError):
    error = "missing_claim"

    def __init__(self, claim):
        description = f"Missing '{claim}' claim"
        super().__init__(description=description)


class InsecureClaimError(JoseError):
    error = "insecure_claim"

    def __init__(self, claim):
        description = f"Insecure claim '{claim}'"
        super().__init__(description=description)


class ExpiredTokenError(JoseError):
    error = "expired_token"
    description = "The token is expired"


class InvalidTokenError(JoseError):
    error = "invalid_token"
    description = "The token is not valid yet"
