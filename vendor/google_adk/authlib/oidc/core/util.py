import hashlib

from authlib.common.encoding import to_bytes
from authlib.common.encoding import urlsafe_b64encode


def create_half_hash(s, alg):
    if alg == "EdDSA":
        hash_alg = hashlib.sha512
    else:
        hash_type = f"sha{alg[2:]}"
        hash_alg = getattr(hashlib, hash_type, None)
        if not hash_alg:
            return None

    data_digest = hash_alg(to_bytes(s)).digest()
    slice_index = int(len(data_digest) / 2)
    return urlsafe_b64encode(data_digest[:slice_index])
