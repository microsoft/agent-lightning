import struct

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash

from authlib.jose.errors import InvalidEncryptionAlgorithmForECDH1PUWithKeyWrappingError
from authlib.jose.rfc7516 import JWEAlgorithmWithTagAwareKeyAgreement
from authlib.jose.rfc7518 import AESAlgorithm
from authlib.jose.rfc7518 import CBCHS2EncAlgorithm
from authlib.jose.rfc7518 import ECKey
from authlib.jose.rfc7518 import u32be_len_input
from authlib.jose.rfc8037 import OKPKey


class ECDH1PUAlgorithm(JWEAlgorithmWithTagAwareKeyAgreement):
    EXTRA_HEADERS = ["epk", "apu", "apv", "skid"]
    ALLOWED_KEY_CLS = (ECKey, OKPKey)

    # https://datatracker.ietf.org/doc/html/draft-madden-jose-ecdh-1pu-04
    def __init__(self, key_size=None):
        if key_size is None:
            self.name = "ECDH-1PU"
            self.description = "ECDH-1PU in the Direct Key Agreement mode"
        else:
            self.name = f"ECDH-1PU+A{key_size}KW"
            self.description = (
                f"ECDH-1PU using Concat KDF and CEK wrapped with A{key_size}KW"
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

    def compute_shared_key(self, shared_key_e, shared_key_s):
        return shared_key_e + shared_key_s

    def compute_fixed_info(self, headers, bit_size, tag):
        if tag is None:
            cctag = b""
        else:
            cctag = u32be_len_input(tag)

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
        pub_info = struct.pack(">I", bit_size) + cctag

        return alg_id + apu_info + apv_info + pub_info

    def compute_derived_key(self, shared_key, fixed_info, bit_size):
        ckdf = ConcatKDFHash(
            algorithm=hashes.SHA256(),
            length=bit_size // 8,
            otherinfo=fixed_info,
            backend=default_backend(),
        )
        return ckdf.derive(shared_key)

    def deliver_at_sender(
        self,
        sender_static_key,
        sender_ephemeral_key,
        recipient_pubkey,
        headers,
        bit_size,
        tag,
    ):
        shared_key_s = sender_static_key.exchange_shared_key(recipient_pubkey)
        shared_key_e = sender_ephemeral_key.exchange_shared_key(recipient_pubkey)
        shared_key = self.compute_shared_key(shared_key_e, shared_key_s)

        fixed_info = self.compute_fixed_info(headers, bit_size, tag)

        return self.compute_derived_key(shared_key, fixed_info, bit_size)

    def deliver_at_recipient(
        self,
        recipient_key,
        sender_static_pubkey,
        sender_ephemeral_pubkey,
        headers,
        bit_size,
        tag,
    ):
        shared_key_s = recipient_key.exchange_shared_key(sender_static_pubkey)
        shared_key_e = recipient_key.exchange_shared_key(sender_ephemeral_pubkey)
        shared_key = self.compute_shared_key(shared_key_e, shared_key_s)

        fixed_info = self.compute_fixed_info(headers, bit_size, tag)

        return self.compute_derived_key(shared_key, fixed_info, bit_size)

    def _generate_ephemeral_key(self, key):
        return key.generate_key(key["crv"], is_private=True)

    def _prepare_headers(self, epk):
        # REQUIRED_JSON_FIELDS contains only public fields
        pub_epk = {k: epk[k] for k in epk.REQUIRED_JSON_FIELDS}
        pub_epk["kty"] = epk.kty
        return {"epk": pub_epk}

    def generate_keys_and_prepare_headers(self, enc_alg, key, sender_key, preset=None):
        if not isinstance(enc_alg, CBCHS2EncAlgorithm):
            raise InvalidEncryptionAlgorithmForECDH1PUWithKeyWrappingError()

        if preset and "epk" in preset:
            epk = preset["epk"]
            h = {}
        else:
            epk = self._generate_ephemeral_key(key)
            h = self._prepare_headers(epk)

        if preset and "cek" in preset:
            cek = preset["cek"]
        else:
            cek = enc_alg.generate_cek()

        return {"epk": epk, "cek": cek, "header": h}

    def _agree_upon_key_at_sender(
        self, enc_alg, headers, key, sender_key, epk, tag=None
    ):
        if self.key_size is None:
            bit_size = enc_alg.CEK_SIZE
        else:
            bit_size = self.key_size

        public_key = key.get_op_key("wrapKey")

        return self.deliver_at_sender(
            sender_key, epk, public_key, headers, bit_size, tag
        )

    def _wrap_cek(self, cek, dk):
        kek = self.aeskw.prepare_key(dk)
        return self.aeskw.wrap_cek(cek, kek)

    def agree_upon_key_and_wrap_cek(
        self, enc_alg, headers, key, sender_key, epk, cek, tag
    ):
        dk = self._agree_upon_key_at_sender(enc_alg, headers, key, sender_key, epk, tag)
        return self._wrap_cek(cek, dk)

    def wrap(self, enc_alg, headers, key, sender_key, preset=None):
        # In this class this method is used in direct key agreement mode only
        if self.key_size is not None:
            raise RuntimeError("Invalid algorithm state detected")

        if preset and "epk" in preset:
            epk = preset["epk"]
            h = {}
        else:
            epk = self._generate_ephemeral_key(key)
            h = self._prepare_headers(epk)

        dk = self._agree_upon_key_at_sender(enc_alg, headers, key, sender_key, epk)

        return {"ek": b"", "cek": dk, "header": h}

    def unwrap(self, enc_alg, ek, headers, key, sender_key, tag=None):
        if "epk" not in headers:
            raise ValueError('Missing "epk" in headers')

        if self.key_size is None:
            bit_size = enc_alg.CEK_SIZE
        else:
            bit_size = self.key_size

        sender_pubkey = sender_key.get_op_key("wrapKey")
        epk = key.import_key(headers["epk"])
        epk_pubkey = epk.get_op_key("wrapKey")
        dk = self.deliver_at_recipient(
            key, sender_pubkey, epk_pubkey, headers, bit_size, tag
        )

        if self.key_size is None:
            return dk

        kek = self.aeskw.prepare_key(dk)
        return self.aeskw.unwrap(enc_alg, ek, headers, kek)


JWE_DRAFT_ALG_ALGORITHMS = [
    ECDH1PUAlgorithm(None),  # ECDH-1PU
    ECDH1PUAlgorithm(128),  # ECDH-1PU+A128KW
    ECDH1PUAlgorithm(192),  # ECDH-1PU+A192KW
    ECDH1PUAlgorithm(256),  # ECDH-1PU+A256KW
]


def register_jwe_alg_draft(cls):
    for alg in JWE_DRAFT_ALG_ALGORITHMS:
        cls.register_algorithm(alg)
