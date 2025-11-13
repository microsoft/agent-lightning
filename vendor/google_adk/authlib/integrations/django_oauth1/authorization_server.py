import logging

from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse

from authlib.common.security import generate_token
from authlib.common.urls import url_encode
from authlib.oauth1 import AuthorizationServer as _AuthorizationServer
from authlib.oauth1 import OAuth1Request
from authlib.oauth1 import TemporaryCredential

from .nonce import exists_nonce_in_cache

log = logging.getLogger(__name__)


class BaseServer(_AuthorizationServer):
    def __init__(self, client_model, token_model, token_generator=None):
        self.client_model = client_model
        self.token_model = token_model

        if token_generator is None:

            def token_generator():
                return {
                    "oauth_token": generate_token(42),
                    "oauth_token_secret": generate_token(48),
                }

        self.token_generator = token_generator
        self._config = getattr(settings, "AUTHLIB_OAUTH1_PROVIDER", {})
        self._nonce_expires_in = self._config.get("nonce_expires_in", 86400)
        methods = self._config.get("signature_methods")
        if methods:
            self.SUPPORTED_SIGNATURE_METHODS = methods

    def get_client_by_id(self, client_id):
        try:
            return self.client_model.objects.get(client_id=client_id)
        except self.client_model.DoesNotExist:
            return None

    def exists_nonce(self, nonce, request):
        return exists_nonce_in_cache(nonce, request, self._nonce_expires_in)

    def create_token_credential(self, request):
        temporary_credential = request.credential
        token = self.token_generator()
        item = self.token_model(
            oauth_token=token["oauth_token"],
            oauth_token_secret=token["oauth_token_secret"],
            user_id=temporary_credential.get_user_id(),
            client_id=temporary_credential.get_client_id(),
        )
        item.save()
        return item

    def check_authorization_request(self, request):
        req = self.create_oauth1_request(request)
        self.validate_authorization_request(req)
        return req

    def create_oauth1_request(self, request):
        if request.method == "POST":
            body = request.POST.dict()
        else:
            body = None
        url = request.build_absolute_uri()
        return OAuth1Request(request.method, url, body, request.headers)

    def handle_response(self, status_code, payload, headers):
        resp = HttpResponse(url_encode(payload), status=status_code)
        for k, v in headers:
            resp[k] = v
        return resp


class CacheAuthorizationServer(BaseServer):
    def __init__(self, client_model, token_model, token_generator=None):
        super().__init__(client_model, token_model, token_generator)
        self._temporary_expires_in = self._config.get(
            "temporary_credential_expires_in", 86400
        )
        self._temporary_credential_key_prefix = self._config.get(
            "temporary_credential_key_prefix", "temporary_credential:"
        )

    def create_temporary_credential(self, request):
        key_prefix = self._temporary_credential_key_prefix
        token = self.token_generator()

        client_id = request.client_id
        redirect_uri = request.redirect_uri
        key = key_prefix + token["oauth_token"]
        token["client_id"] = client_id
        if redirect_uri:
            token["oauth_callback"] = redirect_uri

        cache.set(key, token, timeout=self._temporary_expires_in)
        return TemporaryCredential(token)

    def get_temporary_credential(self, request):
        if not request.token:
            return None

        key_prefix = self._temporary_credential_key_prefix
        key = key_prefix + request.token
        value = cache.get(key)
        if value:
            return TemporaryCredential(value)

    def delete_temporary_credential(self, request):
        if request.token:
            key_prefix = self._temporary_credential_key_prefix
            key = key_prefix + request.token
            cache.delete(key)

    def create_authorization_verifier(self, request):
        key_prefix = self._temporary_credential_key_prefix
        verifier = generate_token(36)
        credential = request.credential
        user = request.user
        key = key_prefix + credential.get_oauth_token()
        credential["oauth_verifier"] = verifier
        credential["user_id"] = user.pk
        cache.set(key, credential, timeout=self._temporary_expires_in)
        return verifier
