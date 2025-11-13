import logging

from flask import Response
from flask import request as flask_req
from werkzeug.utils import import_string

from authlib.common.security import generate_token
from authlib.common.urls import url_encode
from authlib.oauth1 import AuthorizationServer as _AuthorizationServer
from authlib.oauth1 import OAuth1Request

log = logging.getLogger(__name__)


class AuthorizationServer(_AuthorizationServer):
    """Flask implementation of :class:`authlib.rfc5849.AuthorizationServer`.
    Initialize it with Flask app instance, client model class and cache::

        server = AuthorizationServer(app=app, query_client=query_client)
        # or initialize lazily
        server = AuthorizationServer()
        server.init_app(app, query_client=query_client)

    :param app: A Flask app instance
    :param query_client: A function to get client by client_id. The client
        model class MUST implement the methods described by
        :class:`~authlib.oauth1.rfc5849.ClientMixin`.
    :param token_generator: A function to generate token
    """

    def __init__(self, app=None, query_client=None, token_generator=None):
        self.app = app
        self.query_client = query_client
        self.token_generator = token_generator

        self._hooks = {
            "exists_nonce": None,
            "create_temporary_credential": None,
            "get_temporary_credential": None,
            "delete_temporary_credential": None,
            "create_authorization_verifier": None,
            "create_token_credential": None,
        }
        if app is not None:
            self.init_app(app)

    def init_app(self, app, query_client=None, token_generator=None):
        if query_client is not None:
            self.query_client = query_client
        if token_generator is not None:
            self.token_generator = token_generator

        if self.token_generator is None:
            self.token_generator = self.create_token_generator(app)

        methods = app.config.get("OAUTH1_SUPPORTED_SIGNATURE_METHODS")
        if methods and isinstance(methods, (list, tuple)):
            self.SUPPORTED_SIGNATURE_METHODS = methods

        self.app = app

    def register_hook(self, name, func):
        if name not in self._hooks:
            raise ValueError('Invalid "name" of hook')
        self._hooks[name] = func

    def create_token_generator(self, app):
        token_generator = app.config.get("OAUTH1_TOKEN_GENERATOR")

        if isinstance(token_generator, str):
            token_generator = import_string(token_generator)
        else:
            length = app.config.get("OAUTH1_TOKEN_LENGTH", 42)

            def token_generator():
                return generate_token(length)

        secret_generator = app.config.get("OAUTH1_TOKEN_SECRET_GENERATOR")
        if isinstance(secret_generator, str):
            secret_generator = import_string(secret_generator)
        else:
            length = app.config.get("OAUTH1_TOKEN_SECRET_LENGTH", 48)

            def secret_generator():
                return generate_token(length)

        def create_token():
            return {
                "oauth_token": token_generator(),
                "oauth_token_secret": secret_generator(),
            }

        return create_token

    def get_client_by_id(self, client_id):
        return self.query_client(client_id)

    def exists_nonce(self, nonce, request):
        func = self._hooks["exists_nonce"]
        if callable(func):
            timestamp = request.timestamp
            client_id = request.client_id
            token = request.token
            return func(nonce, timestamp, client_id, token)

        raise RuntimeError('"exists_nonce" hook is required.')

    def create_temporary_credential(self, request):
        func = self._hooks["create_temporary_credential"]
        if callable(func):
            token = self.token_generator()
            return func(token, request.client_id, request.redirect_uri)
        raise RuntimeError('"create_temporary_credential" hook is required.')

    def get_temporary_credential(self, request):
        func = self._hooks["get_temporary_credential"]
        if callable(func):
            return func(request.token)

        raise RuntimeError('"get_temporary_credential" hook is required.')

    def delete_temporary_credential(self, request):
        func = self._hooks["delete_temporary_credential"]
        if callable(func):
            return func(request.token)

        raise RuntimeError('"delete_temporary_credential" hook is required.')

    def create_authorization_verifier(self, request):
        func = self._hooks["create_authorization_verifier"]
        if callable(func):
            verifier = generate_token(36)
            func(request.credential, request.user, verifier)
            return verifier

        raise RuntimeError('"create_authorization_verifier" hook is required.')

    def create_token_credential(self, request):
        func = self._hooks["create_token_credential"]
        if callable(func):
            temporary_credential = request.credential
            token = self.token_generator()
            return func(token, temporary_credential)

        raise RuntimeError('"create_token_credential" hook is required.')

    def check_authorization_request(self):
        req = self.create_oauth1_request(None)
        self.validate_authorization_request(req)
        return req

    def create_authorization_response(self, request=None, grant_user=None):
        return super().create_authorization_response(request, grant_user)

    def create_token_response(self, request=None):
        return super().create_token_response(request)

    def create_oauth1_request(self, request):
        if request is None:
            request = flask_req
        if request.method in ("POST", "PUT"):
            body = request.form.to_dict(flat=True)
        else:
            body = None
        return OAuth1Request(request.method, request.url, body, request.headers)

    def handle_response(self, status_code, payload, headers):
        return Response(url_encode(payload), status=status_code, headers=headers)
