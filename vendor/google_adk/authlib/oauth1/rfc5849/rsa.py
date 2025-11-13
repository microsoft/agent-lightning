from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from authlib.common.encoding import to_bytes


def sign_sha1(msg, rsa_private_key):
    key = load_pem_private_key(
        to_bytes(rsa_private_key), password=None, backend=default_backend()
    )
    return key.sign(msg, padding.PKCS1v15(), hashes.SHA1())


def verify_sha1(sig, msg, rsa_public_key):
    key = load_pem_public_key(to_bytes(rsa_public_key), backend=default_backend())
    try:
        key.verify(sig, msg, padding.PKCS1v15(), hashes.SHA1())
        return True
    except InvalidSignature:
        return False
