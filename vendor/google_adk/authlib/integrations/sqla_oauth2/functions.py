import time


def create_query_client_func(session, client_model):
    """Create an ``query_client`` function that can be used in authorization
    server.

    :param session: SQLAlchemy session
    :param client_model: Client model class
    """

    def query_client(client_id):
        q = session.query(client_model)
        return q.filter_by(client_id=client_id).first()

    return query_client


def create_save_token_func(session, token_model):
    """Create an ``save_token`` function that can be used in authorization
    server.

    :param session: SQLAlchemy session
    :param token_model: Token model class
    """

    def save_token(token, request):
        if request.user:
            user_id = request.user.get_user_id()
        else:
            user_id = None
        client = request.client
        item = token_model(client_id=client.client_id, user_id=user_id, **token)
        session.add(item)
        session.commit()

    return save_token


def create_query_token_func(session, token_model):
    """Create an ``query_token`` function for revocation, introspection
    token endpoints.

    :param session: SQLAlchemy session
    :param token_model: Token model class
    """

    def query_token(token, token_type_hint):
        q = session.query(token_model)
        if token_type_hint == "access_token":
            return q.filter_by(access_token=token).first()
        elif token_type_hint == "refresh_token":
            return q.filter_by(refresh_token=token).first()
        # without token_type_hint
        item = q.filter_by(access_token=token).first()
        if item:
            return item
        return q.filter_by(refresh_token=token).first()

    return query_token


def create_revocation_endpoint(session, token_model):
    """Create a revocation endpoint class with SQLAlchemy session
    and token model.

    :param session: SQLAlchemy session
    :param token_model: Token model class
    """
    from authlib.oauth2.rfc7009 import RevocationEndpoint

    query_token = create_query_token_func(session, token_model)

    class _RevocationEndpoint(RevocationEndpoint):
        def query_token(self, token, token_type_hint):
            return query_token(token, token_type_hint)

        def revoke_token(self, token, request):
            now = int(time.time())
            hint = request.form.get("token_type_hint")
            token.access_token_revoked_at = now
            if hint != "access_token":
                token.refresh_token_revoked_at = now
            session.add(token)
            session.commit()

    return _RevocationEndpoint


def create_bearer_token_validator(session, token_model):
    """Create an bearer token validator class with SQLAlchemy session
    and token model.

    :param session: SQLAlchemy session
    :param token_model: Token model class
    """
    from authlib.oauth2.rfc6750 import BearerTokenValidator

    class _BearerTokenValidator(BearerTokenValidator):
        def authenticate_token(self, token_string):
            q = session.query(token_model)
            return q.filter_by(access_token=token_string).first()

    return _BearerTokenValidator
