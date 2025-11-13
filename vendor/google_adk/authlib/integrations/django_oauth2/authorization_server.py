from django.conf import settings
from django.http import HttpResponse
from django.utils.module_loading import import_string

from authlib.common.encoding import json_dumps
from authlib.common.security import generate_token as _generate_token
from authlib.oauth2 import AuthorizationServer as _AuthorizationServer
from authlib.oauth2.rfc6750 import BearerTokenGenerator

from .requests import DjangoJsonRequest
from .requests import DjangoOAuth2Request
from .signals import client_authenticated
from .signals import token_revoked


class AuthorizationServer(_AuthorizationServer):
    """Django implementation of :class:`authlib.oauth2.rfc6749.AuthorizationServer`.
    Initialize it with client model and token model::

        from authlib.integrations.django_oauth2 import AuthorizationServer
        from your_project.models import OAuth2Client, OAuth2Token

        server = AuthorizationServer(OAuth2Client, OAuth2Token)
    """

    def __init__(self, client_model, token_model):
        super().__init__()
        self.client_model = client_model
        self.token_model = token_model
        self.load_config(getattr(settings, "AUTHLIB_OAUTH2_PROVIDER", {}))

    def load_config(self, config):
        self.config = config
        scopes_supported = self.config.get("scopes_supported")
        self.scopes_supported = scopes_supported
        # add default token generator
        self.register_token_generator("default", self.create_bearer_token_generator())

    def query_client(self, client_id):
        """Default method for ``AuthorizationServer.query_client``. Developers MAY
        rewrite this function to meet their own needs.
        """
        try:
            return self.client_model.objects.get(client_id=client_id)
        except self.client_model.DoesNotExist:
            return None

    def save_token(self, token, request):
        """Default method for ``AuthorizationServer.save_token``. Developers MAY
        rewrite this function to meet their own needs.
        """
        client = request.client
        if request.user:
            user_id = request.user.pk
        else:
            user_id = client.user_id
        item = self.token_model(client_id=client.client_id, user_id=user_id, **token)
        item.save()
        return item

    def create_oauth2_request(self, request):
        return DjangoOAuth2Request(request)

    def create_json_request(self, request):
        return DjangoJsonRequest(request)

    def handle_response(self, status_code, payload, headers):
        if isinstance(payload, dict):
            payload = json_dumps(payload)
        resp = HttpResponse(payload, status=status_code)
        for k, v in headers:
            resp[k] = v
        return resp

    def send_signal(self, name, *args, **kwargs):
        if name == "after_authenticate_client":
            client_authenticated.send(*args, sender=self.__class__, **kwargs)
        elif name == "after_revoke_token":
            token_revoked.send(*args, sender=self.__class__, **kwargs)

    def create_bearer_token_generator(self):
        """Default method to create BearerToken generator."""
        conf = self.config.get("access_token_generator", True)
        access_token_generator = create_token_generator(conf, 42)

        conf = self.config.get("refresh_token_generator", False)
        refresh_token_generator = create_token_generator(conf, 48)

        conf = self.config.get("token_expires_in")
        expires_generator = create_token_expires_in_generator(conf)

        return BearerTokenGenerator(
            access_token_generator=access_token_generator,
            refresh_token_generator=refresh_token_generator,
            expires_generator=expires_generator,
        )


def create_token_generator(token_generator_conf, length=42):
    if callable(token_generator_conf):
        return token_generator_conf

    if isinstance(token_generator_conf, str):
        return import_string(token_generator_conf)
    elif token_generator_conf is True:

        def token_generator(*args, **kwargs):
            return _generate_token(length)

        return token_generator


def create_token_expires_in_generator(expires_in_conf=None):
    data = {}
    data.update(BearerTokenGenerator.GRANT_TYPES_EXPIRES_IN)
    if expires_in_conf:
        data.update(expires_in_conf)

    def expires_in(client, grant_type):
        return data.get(grant_type, BearerTokenGenerator.DEFAULT_EXPIRES_IN)

    return expires_in
