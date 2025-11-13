from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.serialization import load_ssh_public_key
from cryptography.x509 import load_pem_x509_certificate

from authlib.common.encoding import to_bytes


def load_pem_key(raw, ssh_type=None, key_type=None, password=None):
    raw = to_bytes(raw)

    if ssh_type and raw.startswith(ssh_type):
        return load_ssh_public_key(raw, backend=default_backend())

    if key_type == "public":
        return load_pem_public_key(raw, backend=default_backend())

    if key_type == "private" or password is not None:
        return load_pem_private_key(raw, password=password, backend=default_backend())

    if b"PUBLIC" in raw:
        return load_pem_public_key(raw, backend=default_backend())

    if b"PRIVATE" in raw:
        return load_pem_private_key(raw, password=password, backend=default_backend())

    if b"CERTIFICATE" in raw:
        cert = load_pem_x509_certificate(raw, default_backend())
        return cert.public_key()

    try:
        return load_pem_private_key(raw, password=password, backend=default_backend())
    except ValueError:
        return load_pem_public_key(raw, backend=default_backend())
