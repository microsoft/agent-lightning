"""authlib.jose.rfc7518.
~~~~~~~~~~~~~~~~~~~~

Cryptographic Algorithms for Cryptographic Algorithms for Content
Encryption per `Section 5`_.

.. _`Section 5`: https://tools.ietf.org/html/rfc7518#section-5
"""

import hashlib
import hmac

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import CBC
from cryptography.hazmat.primitives.ciphers.modes import GCM
from cryptography.hazmat.primitives.padding import PKCS7

from ..rfc7516 import JWEEncAlgorithm
from .util import encode_int


class CBCHS2EncAlgorithm(JWEEncAlgorithm):
    # The IV used is a 128-bit value generated randomly or
    # pseudo-randomly for use in the cipher.
    IV_SIZE = 128

    def __init__(self, key_size, hash_type):
        self.name = f"A{key_size}CBC-HS{hash_type}"
        tpl = "AES_{}_CBC_HMAC_SHA_{} authenticated encryption algorithm"
        self.description = tpl.format(key_size, hash_type)

        # bit length
        self.key_size = key_size
        # byte length
        self.key_len = key_size // 8

        self.CEK_SIZE = key_size * 2
        self.hash_alg = getattr(hashlib, f"sha{hash_type}")

    def _hmac(self, ciphertext, aad, iv, key):
        al = encode_int(len(aad) * 8, 64)
        msg = aad + iv + ciphertext + al
        d = hmac.new(key, msg, self.hash_alg).digest()
        return d[: self.key_len]

    def encrypt(self, msg, aad, iv, key):
        """Key Encryption with AES_CBC_HMAC_SHA2.

        :param msg: text to be encrypt in bytes
        :param aad: additional authenticated data in bytes
        :param iv: initialization vector in bytes
        :param key: encrypted key in bytes
        :return: (ciphertext, iv, tag)
        """
        self.check_iv(iv)
        hkey = key[: self.key_len]
        ekey = key[self.key_len :]

        pad = PKCS7(AES.block_size).padder()
        padded_data = pad.update(msg) + pad.finalize()

        cipher = Cipher(AES(ekey), CBC(iv), backend=default_backend())
        enc = cipher.encryptor()
        ciphertext = enc.update(padded_data) + enc.finalize()
        tag = self._hmac(ciphertext, aad, iv, hkey)
        return ciphertext, tag

    def decrypt(self, ciphertext, aad, iv, tag, key):
        """Key Decryption with AES AES_CBC_HMAC_SHA2.

        :param ciphertext: ciphertext in bytes
        :param aad: additional authenticated data in bytes
        :param iv: initialization vector in bytes
        :param tag: authentication tag in bytes
        :param key: encrypted key in bytes
        :return: message
        """
        self.check_iv(iv)
        hkey = key[: self.key_len]
        dkey = key[self.key_len :]

        _tag = self._hmac(ciphertext, aad, iv, hkey)
        if not hmac.compare_digest(_tag, tag):
            raise InvalidTag()

        cipher = Cipher(AES(dkey), CBC(iv), backend=default_backend())
        d = cipher.decryptor()
        data = d.update(ciphertext) + d.finalize()
        unpad = PKCS7(AES.block_size).unpadder()
        return unpad.update(data) + unpad.finalize()


class GCMEncAlgorithm(JWEEncAlgorithm):
    # Use of an IV of size 96 bits is REQUIRED with this algorithm.
    # https://tools.ietf.org/html/rfc7518#section-5.3
    IV_SIZE = 96

    def __init__(self, key_size):
        self.name = f"A{key_size}GCM"
        self.description = f"AES GCM using {key_size}-bit key"
        self.key_size = key_size
        self.CEK_SIZE = key_size

    def encrypt(self, msg, aad, iv, key):
        """Key Encryption with AES GCM.

        :param msg: text to be encrypt in bytes
        :param aad: additional authenticated data in bytes
        :param iv: initialization vector in bytes
        :param key: encrypted key in bytes
        :return: (ciphertext, iv, tag)
        """
        self.check_iv(iv)
        cipher = Cipher(AES(key), GCM(iv), backend=default_backend())
        enc = cipher.encryptor()
        enc.authenticate_additional_data(aad)
        ciphertext = enc.update(msg) + enc.finalize()
        return ciphertext, enc.tag

    def decrypt(self, ciphertext, aad, iv, tag, key):
        """Key Decryption with AES GCM.

        :param ciphertext: ciphertext in bytes
        :param aad: additional authenticated data in bytes
        :param iv: initialization vector in bytes
        :param tag: authentication tag in bytes
        :param key: encrypted key in bytes
        :return: message
        """
        self.check_iv(iv)
        cipher = Cipher(AES(key), GCM(iv, tag), backend=default_backend())
        d = cipher.decryptor()
        d.authenticate_additional_data(aad)
        return d.update(ciphertext) + d.finalize()


JWE_ENC_ALGORITHMS = [
    CBCHS2EncAlgorithm(128, 256),  # A128CBC-HS256
    CBCHS2EncAlgorithm(192, 384),  # A192CBC-HS384
    CBCHS2EncAlgorithm(256, 512),  # A256CBC-HS512
    GCMEncAlgorithm(128),  # A128GCM
    GCMEncAlgorithm(192),  # A192GCM
    GCMEncAlgorithm(256),  # A256GCM
]
