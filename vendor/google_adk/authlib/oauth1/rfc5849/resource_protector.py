from .base_server import BaseServer
from .errors import InvalidClientError
from .errors import InvalidTokenError
from .errors import MissingRequiredParameterError
from .wrapper import OAuth1Request


class ResourceProtector(BaseServer):
    def validate_request(self, method, uri, body, headers):
        request = OAuth1Request(method, uri, body, headers)

        if not request.client_id:
            raise MissingRequiredParameterError("oauth_consumer_key")

        client = self.get_client_by_id(request.client_id)
        if not client:
            raise InvalidClientError()
        request.client = client

        if not request.token:
            raise MissingRequiredParameterError("oauth_token")

        token = self.get_token_credential(request)
        if not token:
            raise InvalidTokenError()

        request.credential = token
        self.validate_timestamp_and_nonce(request)
        self.validate_oauth_signature(request)
        return request

    def get_token_credential(self, request):
        """Fetch the token credential from data store like a database,
        framework should implement this function.

        :param request: OAuth1Request instance
        :return: Token model instance
        """
        raise NotImplementedError()
