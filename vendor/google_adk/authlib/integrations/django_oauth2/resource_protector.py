import functools

from django.http import JsonResponse

from authlib.oauth2 import OAuth2Error
from authlib.oauth2 import ResourceProtector as _ResourceProtector
from authlib.oauth2.rfc6749 import MissingAuthorizationError
from authlib.oauth2.rfc6750 import BearerTokenValidator as _BearerTokenValidator

from .requests import DjangoJsonRequest
from .signals import token_authenticated


class ResourceProtector(_ResourceProtector):
    def acquire_token(self, request, scopes=None, **kwargs):
        """A method to acquire current valid token with the given scope.

        :param request: Django HTTP request instance
        :param scopes: a list of scope values
        :return: token object
        """
        req = DjangoJsonRequest(request)
        # backward compatibility
        kwargs["scopes"] = scopes
        for claim in kwargs:
            if isinstance(kwargs[claim], str):
                kwargs[claim] = [kwargs[claim]]
        token = self.validate_request(request=req, **kwargs)
        token_authenticated.send(sender=self.__class__, token=token)
        return token

    def __call__(self, scopes=None, optional=False, **kwargs):
        claims = kwargs
        # backward compatibility
        claims["scopes"] = scopes

        def wrapper(f):
            @functools.wraps(f)
            def decorated(request, *args, **kwargs):
                try:
                    token = self.acquire_token(request, **claims)
                    request.oauth_token = token
                except MissingAuthorizationError as error:
                    if optional:
                        request.oauth_token = None
                        return f(request, *args, **kwargs)
                    return return_error_response(error)
                except OAuth2Error as error:
                    return return_error_response(error)
                return f(request, *args, **kwargs)

            return decorated

        return wrapper


class BearerTokenValidator(_BearerTokenValidator):
    def __init__(self, token_model, realm=None, **extra_attributes):
        self.token_model = token_model
        super().__init__(realm, **extra_attributes)

    def authenticate_token(self, token_string):
        try:
            return self.token_model.objects.get(access_token=token_string)
        except self.token_model.DoesNotExist:
            return None


def return_error_response(error):
    body = dict(error.get_body())
    resp = JsonResponse(body, status=error.status_code)
    headers = error.get_headers()
    for k, v in headers:
        resp[k] = v
    return resp
