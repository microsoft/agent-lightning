import time

from authlib.common.security import generate_token
from authlib.jose import jwt


def sign_jwt_bearer_assertion(
    key,
    issuer,
    audience,
    subject=None,
    issued_at=None,
    expires_at=None,
    claims=None,
    header=None,
    **kwargs,
):
    if header is None:
        header = {}
    alg = kwargs.pop("alg", None)
    if alg:
        header["alg"] = alg
    if "alg" not in header:
        raise ValueError("Missing 'alg' in header")

    payload = {"iss": issuer, "aud": audience}

    # subject is not required in Google service
    if subject:
        payload["sub"] = subject

    if not issued_at:
        issued_at = int(time.time())

    expires_in = kwargs.pop("expires_in", 3600)
    if not expires_at:
        expires_at = issued_at + expires_in

    payload["iat"] = issued_at
    payload["exp"] = expires_at

    if claims:
        payload.update(claims)

    return jwt.encode(header, payload, key)


def client_secret_jwt_sign(
    client_secret, client_id, token_endpoint, alg="HS256", claims=None, **kwargs
):
    return _sign(client_secret, client_id, token_endpoint, alg, claims, **kwargs)


def private_key_jwt_sign(
    private_key, client_id, token_endpoint, alg="RS256", claims=None, **kwargs
):
    return _sign(private_key, client_id, token_endpoint, alg, claims, **kwargs)


def _sign(key, client_id, token_endpoint, alg, claims=None, **kwargs):
    # REQUIRED. Issuer. This MUST contain the client_id of the OAuth Client.
    issuer = client_id
    # REQUIRED. Subject. This MUST contain the client_id of the OAuth Client.
    subject = client_id
    # The Audience SHOULD be the URL of the Authorization Server's Token Endpoint.
    audience = token_endpoint

    # jti is required
    if claims is None:
        claims = {}
    if "jti" not in claims:
        claims["jti"] = generate_token(36)

    return sign_jwt_bearer_assertion(
        key=key,
        issuer=issuer,
        audience=audience,
        subject=subject,
        claims=claims,
        alg=alg,
        **kwargs,
    )
