class JWSAlgorithm:
    """Interface for JWS algorithm. JWA specification (RFC7518) SHOULD
    implement the algorithms for JWS with this base implementation.
    """

    name = None
    description = None
    algorithm_type = "JWS"
    algorithm_location = "alg"

    def prepare_key(self, raw_data):
        """Prepare key for signing and verifying signature."""
        raise NotImplementedError()

    def sign(self, msg, key):
        """Sign the text msg with a private/sign key.

        :param msg: message bytes to be signed
        :param key: private key to sign the message
        :return: bytes
        """
        raise NotImplementedError

    def verify(self, msg, sig, key):
        """Verify the signature of text msg with a public/verify key.

        :param msg: message bytes to be signed
        :param sig: result signature to be compared
        :param key: public key to verify the signature
        :return: boolean
        """
        raise NotImplementedError


class JWSHeader(dict):
    """Header object for JWS. It combine the protected header and unprotected
    header together. JWSHeader itself is a dict of the combined dict. e.g.

        >>> protected = {"alg": "HS256"}
        >>> header = {"kid": "a"}
        >>> jws_header = JWSHeader(protected, header)
        >>> print(jws_header)
        {'alg': 'HS256', 'kid': 'a'}
        >>> jws_header.protected == protected
        >>> jws_header.header == header

    :param protected: dict of protected header
    :param header: dict of unprotected header
    """

    def __init__(self, protected, header):
        obj = {}
        if header:
            obj.update(header)
        if protected:
            obj.update(protected)
        super().__init__(obj)
        self.protected = protected
        self.header = header

    @classmethod
    def from_dict(cls, obj):
        if isinstance(obj, cls):
            return obj
        return cls(obj.get("protected"), obj.get("header"))


class JWSObject(dict):
    """A dict instance to represent a JWS object."""

    def __init__(self, header, payload, type="compact"):
        super().__init__(
            header=header,
            payload=payload,
        )
        self.header = header
        self.payload = payload
        self.type = type

    @property
    def headers(self):
        """Alias of ``header`` for JSON typed JWS."""
        if self.type == "json":
            return self["header"]
