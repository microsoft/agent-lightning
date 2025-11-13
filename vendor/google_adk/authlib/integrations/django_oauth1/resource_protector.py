import functools

from django.conf import settings
from django.http import JsonResponse

from authlib.oauth1 import ResourceProtector as _ResourceProtector
from authlib.oauth1.errors import OAuth1Error

from .nonce import exists_nonce_in_cache


class ResourceProtector(_ResourceProtector):
    def __init__(self, client_model, token_model):
        self.client_model = client_model
        self.token_model = token_model

        config = getattr(settings, "AUTHLIB_OAUTH1_PROVIDER", {})
        methods = config.get("signature_methods", [])
        if methods and isinstance(methods, (list, tuple)):
            self.SUPPORTED_SIGNATURE_METHODS = methods

        self._nonce_expires_in = config.get("nonce_expires_in", 86400)

    def get_client_by_id(self, client_id):
        try:
            return self.client_model.objects.get(client_id=client_id)
        except self.client_model.DoesNotExist:
            return None

    def get_token_credential(self, request):
        try:
            return self.token_model.objects.get(
                client_id=request.client_id, oauth_token=request.token
            )
        except self.token_model.DoesNotExist:
            return None

    def exists_nonce(self, nonce, request):
        return exists_nonce_in_cache(nonce, request, self._nonce_expires_in)

    def acquire_credential(self, request):
        if request.method in ["POST", "PUT"]:
            body = request.POST.dict()
        else:
            body = None

        url = request.build_absolute_uri()
        req = self.validate_request(request.method, url, body, request.headers)
        return req.credential

    def __call__(self, realm=None):
        def wrapper(f):
            @functools.wraps(f)
            def decorated(request, *args, **kwargs):
                try:
                    credential = self.acquire_credential(request)
                    request.oauth1_credential = credential
                except OAuth1Error as error:
                    body = dict(error.get_body())
                    resp = JsonResponse(body, status=error.status_code)
                    resp["Cache-Control"] = "no-store"
                    resp["Pragma"] = "no-cache"
                    return resp
                return f(request, *args, **kwargs)

            return decorated

        return wrapper
