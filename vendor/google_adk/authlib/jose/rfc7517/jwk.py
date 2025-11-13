from authlib.common.encoding import json_loads

from ._cryptography_key import load_pem_key
from .key_set import KeySet


class JsonWebKey:
    JWK_KEY_CLS = {}

    @classmethod
    def generate_key(cls, kty, crv_or_size, options=None, is_private=False):
        """Generate a Key with the given key type, curve name or bit size.

        :param kty: string of ``oct``, ``RSA``, ``EC``, ``OKP``
        :param crv_or_size: curve name or bit size
        :param options: a dict of other options for Key
        :param is_private: create a private key or public key
        :return: Key instance
        """
        key_cls = cls.JWK_KEY_CLS[kty]
        return key_cls.generate_key(crv_or_size, options, is_private)

    @classmethod
    def import_key(cls, raw, options=None):
        """Import a Key from bytes, string, PEM or dict.

        :return: Key instance
        """
        kty = None
        if options is not None:
            kty = options.get("kty")

        if kty is None and isinstance(raw, dict):
            kty = raw.get("kty")

        if kty is None:
            raw_key = load_pem_key(raw)
            for _kty in cls.JWK_KEY_CLS:
                key_cls = cls.JWK_KEY_CLS[_kty]
                if key_cls.validate_raw_key(raw_key):
                    return key_cls.import_key(raw_key, options)

        key_cls = cls.JWK_KEY_CLS[kty]
        return key_cls.import_key(raw, options)

    @classmethod
    def import_key_set(cls, raw):
        """Import KeySet from string, dict or a list of keys.

        :return: KeySet instance
        """
        raw = _transform_raw_key(raw)
        if isinstance(raw, dict) and "keys" in raw:
            keys = raw.get("keys")
            return KeySet([cls.import_key(k) for k in keys])
        raise ValueError("Invalid key set format")


def _transform_raw_key(raw):
    if isinstance(raw, str) and raw.startswith("{") and raw.endswith("}"):
        return json_loads(raw)
    elif isinstance(raw, (tuple, list)):
        return {"keys": raw}
    return raw
