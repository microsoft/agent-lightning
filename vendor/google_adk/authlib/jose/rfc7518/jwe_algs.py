import os
import struct

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import GCM
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap
from cryptography.hazmat.primitives.keywrap import aes_key_wrap

from authlib.common.encoding import to_bytes
from authlib.common.encoding import to_native
from authlib.common.encoding import urlsafe_b64decode
from authlib.common.encoding import urlsafe_b64encode
from authlib.jose.rfc7516 import JWEAlgorithm

from .ec_key import ECKey
from .oct_key import OctKey
from .rsa_key import RSAKey


class DirectAlgorithm(JWEAlgorithm):
    name = "dir"
    description = "Direct use of a shared symmetric key"

    def prepare_key(self, raw_data):
        return OctKey.import_key(raw_data)

    def generate_preset(self, enc_alg, key):
        return {}

    def wrap(self, enc_alg, headers, key, preset=None):
        cek = key.get_op_key("encrypt")
        if len(cek) * 8 != enc_alg.CEK_SIZE:
            raise ValueError('Invalid "cek" length')
        return {"ek": b"", "cek": cek}

    def unwrap(self, enc_alg, ek, headers, key):
        cek = key.get_op_key("decrypt")
        if len(cek) * 8 != enc_alg.CEK_SIZE:
            raise ValueError('Invalid "cek" length')
        return cek


class RSAAlgorithm(JWEAlgorithm):
    #: A key of size 2048 bits or larger MUST be used with these algorithms
    #: RSA1_5, RSA-OAEP, RSA-OAEP-256
    key_size = 2048

    def __init__(self, name, description, pad_fn):
        self.name = name
        self.description = description
        self.padding = pad_fn

    def prepare_key(self, raw_data):
        return RSAKey.import_key(raw_data)

    def generate_preset(self, enc_alg, key):
        cek = enc_alg.generate_cek()
        return {"cek": cek}

    def wrap(self, enc_alg, headers, key, preset=None):
        if preset and "cek" in preset:
            cek = preset["cek"]
        else:
            cek = enc_alg.generate_cek()

        op_key = key.get_op_key("wrapKey")
        if op_key.key_size < self.key_size:
            raise ValueError("A key of size 2048 bits or larger MUST be used")
        ek = op_key.encrypt(cek, self.padding)
        return {"ek": ek, "cek": cek}

    def unwrap(self, enc_alg, ek, headers, key):
        # it will raise ValueError if failed
        op_key = key.get_op_key("unwrapKey")
        cek = op_key.decrypt(ek, self.padding)
        if len(cek) * 8 != enc_alg.CEK_SIZE:
            raise ValueError('Invalid "cek" length')
        return cek


class AESAlgorithm(JWEAlgorithm):
    def __init__(self, key_size):
        self.name = f"A{key_size}KW"
        self.description = f"AES Key Wrap using {key_size}-bit key"
        self.key_size = key_size

    def prepare_key(self, raw_data):
        return OctKey.import_key(raw_data)

    def generate_preset(self, enc_alg, key):
        cek = enc_alg.generate_cek()
        return {"cek": cek}

    def _check_key(self, key):
        if len(key) * 8 != self.key_size:
            raise ValueError(f"A key of size {self.key_size} bits is required.")

    def wrap_cek(self, cek, key):
        op_key = key.get_op_key("wrapKey")
        self._check_key(op_key)
        ek = aes_key_wrap(op_key, cek, default_backend())
        return {"ek": ek, "cek": cek}

    def wrap(self, enc_alg, headers, key, preset=None):
        if preset and "cek" in preset:
            cek = preset["cek"]
        else:
            cek = enc_alg.generate_cek()
        return self.wrap_cek(cek, key)

    def unwrap(self, enc_alg, ek, headers, key):
        op_key = key.get_op_key("unwrapKey")
        self._check_key(op_key)
        cek = aes_key_unwrap(op_key, ek, default_backend())
        if len(cek) * 8 != enc_alg.CEK_SIZE:
            raise ValueError('Invalid "cek" length')
        return cek


class AESGCMAlgorithm(JWEAlgorithm):
    EXTRA_HEADERS = frozenset(["iv", "tag"])

    def __init__(self, key_size):
        self.name = f"A{key_size}GCMKW"
        self.description = f"Key wrapping with AES GCM using {key_size}-bit key"
        self.key_size = key_size

    def prepare_key(self, raw_data):
        return OctKey.import_key(raw_data)

    def generate_preset(self, enc_alg, key):
        cek = enc_alg.generate_cek()
        return {"cek": cek}

    def _check_key(self, key):
        if len(key) * 8 != self.key_size:
            raise ValueError(f"A key of size {self.key_size} bits is required.")

    def wrap(self, enc_alg, headers, key, preset=None):
        if preset and "cek" in preset:
            cek = preset["cek"]
        else:
            cek = enc_alg.generate_cek()

        op_key = key.get_op_key("wrapKey")
        self._check_key(op_key)

        #: https://tools.ietf.org/html/rfc7518#section-4.7.1.1
        #: The "iv" (initialization vector) Header Parameter value is the
        #: base64url-encoded representation of the 96-bit IV value
        iv_size = 96
        iv = os.urandom(iv_size // 8)

        cipher = Cipher(AES(op_key), GCM(iv), backend=default_backend())
        enc = cipher.encryptor()
        ek = enc.update(cek) + enc.finalize()

        h = {
            "iv": to_native(urlsafe_b64encode(iv)),
            "tag": to_native(urlsafe_b64encode(enc.tag)),
        }
        return {"ek": ek, "cek": cek, "header": h}

    def unwrap(self, enc_alg, ek, headers, key):
        op_key = key.get_op_key("unwrapKey")
        self._check_key(op_key)

        iv = headers.get("iv")
        if not iv:
            raise ValueError('Missing "iv" in headers')

        tag = headers.get("tag")
        if not tag:
            raise ValueError('Missing "tag" in headers')

        iv = urlsafe_b64decode(to_bytes(iv))
        tag = urlsafe_b64decode(to_bytes(tag))

        cipher = Cipher(AES(op_key), GCM(iv, tag), backend=default_backend())
        d = cipher.decryptor()
        cek = d.update(ek) + d.finalize()
        if len(cek) * 8 != enc_alg.CEK_SIZE:
            raise ValueError('Invalid "cek" length')
        return cek


class ECDHESAlgorithm(JWEAlgorithm):
    EXTRA_HEADERS = ["epk", "apu", "apv"]
    ALLOWED_KEY_CLS = ECKey

    # https://tools.ietf.org/html/rfc7518#section-4.6
    def __init__(self, key_size=None):
        if key_size is None:
            self.name = "ECDH-ES"
            self.description = "ECDH-ES in the Direct Key Agreement mode"
        else:
            self.name = f"ECDH-ES+A{key_size}KW"
            self.description = (
                f"ECDH-ES using Concat KDF and CEK wrapped with A{key_size}KW"
            )
        self.key_size = key_size
        self.aeskw = AESAlgorithm(key_size)

    def prepare_key(self, raw_data):
        if isinstance(raw_data, self.ALLOWED_KEY_CLS):
            return raw_data
        return ECKey.import_key(raw_data)

    def generate_preset(self, enc_alg, key):
        epk = self._generate_ephemeral_key(key)
        h = self._prepare_headers(epk)
        preset = {"epk": epk, "header": h}
        if self.key_size is not None:
            cek = enc_alg.generate_cek()
            preset["cek"] = cek
        return preset

    def compute_fixed_info(self, headers, bit_size):
        # AlgorithmID
        if self.key_size is None:
            alg_id = u32be_len_input(headers["enc"])
        else:
            alg_id = u32be_len_input(headers["alg"])

        # PartyUInfo
        apu_info = u32be_len_input(headers.get("apu"), True)

        # PartyVInfo
        apv_info = u32be_len_input(headers.get("apv"), True)

        # SuppPubInfo
        pub_info = struct.pack(">I", bit_size)

        return alg_id + apu_info + apv_info + pub_info

    def compute_derived_key(self, shared_key, fixed_info, bit_size):
        ckdf = ConcatKDFHash(
            algorithm=hashes.SHA256(),
            length=bit_size // 8,
            otherinfo=fixed_info,
            backend=default_backend(),
        )
        return ckdf.derive(shared_key)

    def deliver(self, key, pubkey, headers, bit_size):
        shared_key = key.exchange_shared_key(pubkey)
        fixed_info = self.compute_fixed_info(headers, bit_size)
        return self.compute_derived_key(shared_key, fixed_info, bit_size)

    def _generate_ephemeral_key(self, key):
        return key.generate_key(key["crv"], is_private=True)

    def _prepare_headers(self, epk):
        # REQUIRED_JSON_FIELDS contains only public fields
        pub_epk = {k: epk[k] for k in epk.REQUIRED_JSON_FIELDS}
        pub_epk["kty"] = epk.kty
        return {"epk": pub_epk}

    def wrap(self, enc_alg, headers, key, preset=None):
        if self.key_size is None:
            bit_size = enc_alg.CEK_SIZE
        else:
            bit_size = self.key_size

        if preset and "epk" in preset:
            epk = preset["epk"]
            h = {}
        else:
            epk = self._generate_ephemeral_key(key)
            h = self._prepare_headers(epk)

        public_key = key.get_op_key("wrapKey")
        dk = self.deliver(epk, public_key, headers, bit_size)

        if self.key_size is None:
            return {"ek": b"", "cek": dk, "header": h}

        if preset and "cek" in preset:
            preset_for_kw = {"cek": preset["cek"]}
        else:
            preset_for_kw = None

        kek = self.aeskw.prepare_key(dk)
        rv = self.aeskw.wrap(enc_alg, headers, kek, preset_for_kw)
        rv["header"] = h
        return rv

    def unwrap(self, enc_alg, ek, headers, key):
        if "epk" not in headers:
            raise ValueError('Missing "epk" in headers')

        if self.key_size is None:
            bit_size = enc_alg.CEK_SIZE
        else:
            bit_size = self.key_size

        epk = key.import_key(headers["epk"])
        public_key = epk.get_op_key("wrapKey")
        dk = self.deliver(key, public_key, headers, bit_size)

        if self.key_size is None:
            return dk

        kek = self.aeskw.prepare_key(dk)
        return self.aeskw.unwrap(enc_alg, ek, headers, kek)


def u32be_len_input(s, base64=False):
    if not s:
        return b"\x00\x00\x00\x00"
    if base64:
        s = urlsafe_b64decode(to_bytes(s))
    else:
        s = to_bytes(s)
    return struct.pack(">I", len(s)) + s


JWE_ALG_ALGORITHMS = [
    DirectAlgorithm(),  # dir
    RSAAlgorithm("RSA1_5", "RSAES-PKCS1-v1_5", padding.PKCS1v15()),
    RSAAlgorithm(
        "RSA-OAEP",
        "RSAES OAEP using default parameters",
        padding.OAEP(padding.MGF1(hashes.SHA1()), hashes.SHA1(), None),
    ),
    RSAAlgorithm(
        "RSA-OAEP-256",
        "RSAES OAEP using SHA-256 and MGF1 with SHA-256",
        padding.OAEP(padding.MGF1(hashes.SHA256()), hashes.SHA256(), None),
    ),
    AESAlgorithm(128),  # A128KW
    AESAlgorithm(192),  # A192KW
    AESAlgorithm(256),  # A256KW
    AESGCMAlgorithm(128),  # A128GCMKW
    AESGCMAlgorithm(192),  # A192GCMKW
    AESGCMAlgorithm(256),  # A256GCMKW
    ECDHESAlgorithm(None),  # ECDH-ES
    ECDHESAlgorithm(128),  # ECDH-ES+A128KW
    ECDHESAlgorithm(192),  # ECDH-ES+A192KW
    ECDHESAlgorithm(256),  # ECDH-ES+A256KW
]

# 'PBES2-HS256+A128KW': '',
# 'PBES2-HS384+A192KW': '',
# 'PBES2-HS512+A256KW': '',
