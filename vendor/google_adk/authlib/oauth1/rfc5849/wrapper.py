from urllib.request import parse_http_list
from urllib.request import parse_keqv_list

from authlib.common.urls import extract_params
from authlib.common.urls import url_decode
from authlib.common.urls import urlparse

from .errors import DuplicatedOAuthProtocolParameterError
from .errors import InsecureTransportError
from .signature import SIGNATURE_TYPE_BODY
from .signature import SIGNATURE_TYPE_HEADER
from .signature import SIGNATURE_TYPE_QUERY
from .util import unescape


class OAuth1Request:
    def __init__(self, method, uri, body=None, headers=None):
        InsecureTransportError.check(uri)
        self.method = method
        self.uri = uri
        self.body = body
        self.headers = headers or {}

        # states namespaces
        self.client = None
        self.credential = None
        self.user = None

        self.query = urlparse.urlparse(uri).query
        self.query_params = url_decode(self.query)
        self.body_params = extract_params(body) or []

        self.auth_params, self.realm = _parse_authorization_header(headers)
        self.signature_type, self.oauth_params = _parse_oauth_params(
            self.query_params, self.body_params, self.auth_params
        )

        params = []
        params.extend(self.query_params)
        params.extend(self.body_params)
        params.extend(self.auth_params)
        self.params = params

    @property
    def client_id(self):
        return self.oauth_params.get("oauth_consumer_key")

    @property
    def client_secret(self):
        if self.client:
            return self.client.get_client_secret()

    @property
    def rsa_public_key(self):
        if self.client:
            return self.client.get_rsa_public_key()

    @property
    def timestamp(self):
        return self.oauth_params.get("oauth_timestamp")

    @property
    def redirect_uri(self):
        return self.oauth_params.get("oauth_callback")

    @property
    def signature(self):
        return self.oauth_params.get("oauth_signature")

    @property
    def signature_method(self):
        return self.oauth_params.get("oauth_signature_method")

    @property
    def token(self):
        return self.oauth_params.get("oauth_token")

    @property
    def token_secret(self):
        if self.credential:
            return self.credential.get_oauth_token_secret()


def _filter_oauth(params):
    for k, v in params:
        if k.startswith("oauth_"):
            yield (k, v)


def _parse_authorization_header(headers):
    """Parse an OAuth authorization header into a list of 2-tuples."""
    authorization_header = headers.get("Authorization")
    if not authorization_header:
        return [], None

    auth_scheme = "oauth "
    if authorization_header.lower().startswith(auth_scheme):
        items = parse_http_list(authorization_header[len(auth_scheme) :])
        try:
            items = parse_keqv_list(items).items()
            auth_params = [(unescape(k), unescape(v)) for k, v in items]
            realm = dict(auth_params).get("realm")
            return auth_params, realm
        except (IndexError, ValueError):
            pass
    raise ValueError("Malformed authorization header")


def _parse_oauth_params(query_params, body_params, auth_params):
    oauth_params_set = [
        (SIGNATURE_TYPE_QUERY, list(_filter_oauth(query_params))),
        (SIGNATURE_TYPE_BODY, list(_filter_oauth(body_params))),
        (SIGNATURE_TYPE_HEADER, list(_filter_oauth(auth_params))),
    ]
    oauth_params_set = [params for params in oauth_params_set if params[1]]
    if len(oauth_params_set) > 1:
        found_types = [p[0] for p in oauth_params_set]
        raise DuplicatedOAuthProtocolParameterError(
            '"oauth_" params must come from only 1 signature type '
            "but were found in {}".format(",".join(found_types))
        )

    if oauth_params_set:
        signature_type = oauth_params_set[0][0]
        oauth_params = dict(oauth_params_set[0][1])
    else:
        signature_type = None
        oauth_params = {}
    return signature_type, oauth_params
