"""authlib.jose.draft.
~~~~~~~~~~~~~~~~~~~~

Content Encryption per `Section 4`_.

.. _`Section 4`: https://datatracker.ietf.org/doc/html/draft-amringer-jose-chacha-02#section-4
"""

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from authlib.jose.rfc7516 import JWEEncAlgorithm


class C20PEncAlgorithm(JWEEncAlgorithm):
    # Use of an IV of size 96 bits is REQUIRED with this algorithm.
    # https://datatracker.ietf.org/doc/html/draft-amringer-jose-chacha-02#section-4.1
    IV_SIZE = 96

    def __init__(self, key_size):
        self.name = "C20P"
        self.description = "ChaCha20-Poly1305"
        self.key_size = key_size
        self.CEK_SIZE = key_size

    def encrypt(self, msg, aad, iv, key):
        """Content Encryption with AEAD_CHACHA20_POLY1305.

        :param msg: text to be encrypt in bytes
        :param aad: additional authenticated data in bytes
        :param iv: initialization vector in bytes
        :param key: encrypted key in bytes
        :return: (ciphertext, tag)
        """
        self.check_iv(iv)
        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(iv, msg, aad)
        return ciphertext[:-16], ciphertext[-16:]

    def decrypt(self, ciphertext, aad, iv, tag, key):
        """Content Decryption with AEAD_CHACHA20_POLY1305.

        :param ciphertext: ciphertext in bytes
        :param aad: additional authenticated data in bytes
        :param iv: initialization vector in bytes
        :param tag: authentication tag in bytes
        :param key: encrypted key in bytes
        :return: message
        """
        self.check_iv(iv)
        chacha = ChaCha20Poly1305(key)
        return chacha.decrypt(iv, ciphertext + tag, aad)
