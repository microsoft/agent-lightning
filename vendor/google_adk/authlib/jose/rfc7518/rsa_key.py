from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKeyWithSerialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateNumbers
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from cryptography.hazmat.primitives.asymmetric.rsa import rsa_crt_dmp1
from cryptography.hazmat.primitives.asymmetric.rsa import rsa_crt_dmq1
from cryptography.hazmat.primitives.asymmetric.rsa import rsa_crt_iqmp
from cryptography.hazmat.primitives.asymmetric.rsa import rsa_recover_prime_factors

from authlib.common.encoding import base64_to_int
from authlib.common.encoding import int_to_base64

from ..rfc7517 import AsymmetricKey


class RSAKey(AsymmetricKey):
    """Key class of the ``RSA`` key type."""

    kty = "RSA"
    PUBLIC_KEY_CLS = RSAPublicKey
    PRIVATE_KEY_CLS = RSAPrivateKeyWithSerialization

    PUBLIC_KEY_FIELDS = ["e", "n"]
    PRIVATE_KEY_FIELDS = ["d", "dp", "dq", "e", "n", "p", "q", "qi"]
    REQUIRED_JSON_FIELDS = ["e", "n"]
    SSH_PUBLIC_PREFIX = b"ssh-rsa"

    def dumps_private_key(self):
        numbers = self.private_key.private_numbers()
        return {
            "n": int_to_base64(numbers.public_numbers.n),
            "e": int_to_base64(numbers.public_numbers.e),
            "d": int_to_base64(numbers.d),
            "p": int_to_base64(numbers.p),
            "q": int_to_base64(numbers.q),
            "dp": int_to_base64(numbers.dmp1),
            "dq": int_to_base64(numbers.dmq1),
            "qi": int_to_base64(numbers.iqmp),
        }

    def dumps_public_key(self):
        numbers = self.public_key.public_numbers()
        return {"n": int_to_base64(numbers.n), "e": int_to_base64(numbers.e)}

    def load_private_key(self):
        obj = self._dict_data

        if "oth" in obj:  # pragma: no cover
            # https://tools.ietf.org/html/rfc7518#section-6.3.2.7
            raise ValueError('"oth" is not supported yet')

        public_numbers = RSAPublicNumbers(
            base64_to_int(obj["e"]), base64_to_int(obj["n"])
        )

        if has_all_prime_factors(obj):
            numbers = RSAPrivateNumbers(
                d=base64_to_int(obj["d"]),
                p=base64_to_int(obj["p"]),
                q=base64_to_int(obj["q"]),
                dmp1=base64_to_int(obj["dp"]),
                dmq1=base64_to_int(obj["dq"]),
                iqmp=base64_to_int(obj["qi"]),
                public_numbers=public_numbers,
            )
        else:
            d = base64_to_int(obj["d"])
            p, q = rsa_recover_prime_factors(public_numbers.n, d, public_numbers.e)
            numbers = RSAPrivateNumbers(
                d=d,
                p=p,
                q=q,
                dmp1=rsa_crt_dmp1(d, p),
                dmq1=rsa_crt_dmq1(d, q),
                iqmp=rsa_crt_iqmp(p, q),
                public_numbers=public_numbers,
            )

        return numbers.private_key(default_backend())

    def load_public_key(self):
        numbers = RSAPublicNumbers(
            base64_to_int(self._dict_data["e"]), base64_to_int(self._dict_data["n"])
        )
        return numbers.public_key(default_backend())

    @classmethod
    def generate_key(cls, key_size=2048, options=None, is_private=False) -> "RSAKey":
        if key_size < 512:
            raise ValueError("key_size must not be less than 512")
        if key_size % 8 != 0:
            raise ValueError("Invalid key_size for RSAKey")
        raw_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend(),
        )
        if not is_private:
            raw_key = raw_key.public_key()
        return cls.import_key(raw_key, options=options)

    @classmethod
    def import_dict_key(cls, raw, options=None):
        cls.check_required_fields(raw)
        key = cls(options=options)
        key._dict_data = raw
        if "d" in raw and not has_all_prime_factors(raw):
            # reload dict key
            key.load_raw_key()
            key.load_dict_key()
        return key


def has_all_prime_factors(obj):
    props = ["p", "q", "dp", "dq", "qi"]
    props_found = [prop in obj for prop in props]
    if all(props_found):
        return True

    if any(props_found):
        raise ValueError(
            "RSA key must include all parameters if any are present besides d"
        )

    return False
