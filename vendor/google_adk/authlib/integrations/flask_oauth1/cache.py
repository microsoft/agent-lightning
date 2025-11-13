from authlib.oauth1 import TemporaryCredential


def register_temporary_credential_hooks(
    authorization_server, cache, key_prefix="temporary_credential:"
):
    """Register temporary credential related hooks to authorization server.

    :param authorization_server: AuthorizationServer instance
    :param cache: Cache instance
    :param key_prefix: key prefix for temporary credential
    """

    def create_temporary_credential(token, client_id, redirect_uri):
        key = key_prefix + token["oauth_token"]
        token["client_id"] = client_id
        if redirect_uri:
            token["oauth_callback"] = redirect_uri

        cache.set(key, token, timeout=86400)  # cache for one day
        return TemporaryCredential(token)

    def get_temporary_credential(oauth_token):
        if not oauth_token:
            return None
        key = key_prefix + oauth_token
        value = cache.get(key)
        if value:
            return TemporaryCredential(value)

    def delete_temporary_credential(oauth_token):
        if oauth_token:
            key = key_prefix + oauth_token
            cache.delete(key)

    def create_authorization_verifier(credential, grant_user, verifier):
        key = key_prefix + credential.get_oauth_token()
        credential["oauth_verifier"] = verifier
        credential["user_id"] = grant_user.get_user_id()
        cache.set(key, credential, timeout=86400)
        return credential

    authorization_server.register_hook(
        "create_temporary_credential", create_temporary_credential
    )
    authorization_server.register_hook(
        "get_temporary_credential", get_temporary_credential
    )
    authorization_server.register_hook(
        "delete_temporary_credential", delete_temporary_credential
    )
    authorization_server.register_hook(
        "create_authorization_verifier", create_authorization_verifier
    )


def create_exists_nonce_func(cache, key_prefix="nonce:", expires=86400):
    """Create an ``exists_nonce`` function that can be used in hooks and
    resource protector.

    :param cache: Cache instance
    :param key_prefix: key prefix for temporary credential
    :param expires: Expire time for nonce
    """

    def exists_nonce(nonce, timestamp, client_id, oauth_token):
        key = f"{key_prefix}{nonce}-{timestamp}-{client_id}"
        if oauth_token:
            key = f"{key}-{oauth_token}"
        rv = cache.has(key)
        cache.set(key, 1, timeout=expires)
        return rv

    return exists_nonce


def register_nonce_hooks(
    authorization_server, cache, key_prefix="nonce:", expires=86400
):
    """Register nonce related hooks to authorization server.

    :param authorization_server: AuthorizationServer instance
    :param cache: Cache instance
    :param key_prefix: key prefix for temporary credential
    :param expires: Expire time for nonce
    """
    exists_nonce = create_exists_nonce_func(cache, key_prefix, expires)
    authorization_server.register_hook("exists_nonce", exists_nonce)
