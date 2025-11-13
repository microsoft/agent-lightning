import secrets

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text

from authlib.common.encoding import json_dumps
from authlib.common.encoding import json_loads
from authlib.oauth2.rfc6749 import ClientMixin
from authlib.oauth2.rfc6749 import list_to_scope
from authlib.oauth2.rfc6749 import scope_to_list


class OAuth2ClientMixin(ClientMixin):
    client_id = Column(String(48), index=True)
    client_secret = Column(String(120))
    client_id_issued_at = Column(Integer, nullable=False, default=0)
    client_secret_expires_at = Column(Integer, nullable=False, default=0)
    _client_metadata = Column("client_metadata", Text)

    @property
    def client_info(self):
        """Implementation for Client Info in OAuth 2.0 Dynamic Client
        Registration Protocol via `Section 3.2.1`_.

        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc7591#section-3.2.1
        """
        return dict(
            client_id=self.client_id,
            client_secret=self.client_secret,
            client_id_issued_at=self.client_id_issued_at,
            client_secret_expires_at=self.client_secret_expires_at,
        )

    @property
    def client_metadata(self):
        if "client_metadata" in self.__dict__:
            return self.__dict__["client_metadata"]
        if self._client_metadata:
            data = json_loads(self._client_metadata)
            self.__dict__["client_metadata"] = data
            return data
        return {}

    def set_client_metadata(self, value):
        self._client_metadata = json_dumps(value)
        if "client_metadata" in self.__dict__:
            del self.__dict__["client_metadata"]

    @property
    def redirect_uris(self):
        return self.client_metadata.get("redirect_uris", [])

    @property
    def token_endpoint_auth_method(self):
        return self.client_metadata.get(
            "token_endpoint_auth_method", "client_secret_basic"
        )

    @property
    def grant_types(self):
        return self.client_metadata.get("grant_types", [])

    @property
    def response_types(self):
        return self.client_metadata.get("response_types", [])

    @property
    def client_name(self):
        return self.client_metadata.get("client_name")

    @property
    def client_uri(self):
        return self.client_metadata.get("client_uri")

    @property
    def logo_uri(self):
        return self.client_metadata.get("logo_uri")

    @property
    def scope(self):
        return self.client_metadata.get("scope", "")

    @property
    def contacts(self):
        return self.client_metadata.get("contacts", [])

    @property
    def tos_uri(self):
        return self.client_metadata.get("tos_uri")

    @property
    def policy_uri(self):
        return self.client_metadata.get("policy_uri")

    @property
    def jwks_uri(self):
        return self.client_metadata.get("jwks_uri")

    @property
    def jwks(self):
        return self.client_metadata.get("jwks", [])

    @property
    def software_id(self):
        return self.client_metadata.get("software_id")

    @property
    def software_version(self):
        return self.client_metadata.get("software_version")

    @property
    def id_token_signed_response_alg(self):
        return self.client_metadata.get("id_token_signed_response_alg")

    def get_client_id(self):
        return self.client_id

    def get_default_redirect_uri(self):
        if self.redirect_uris:
            return self.redirect_uris[0]

    def get_allowed_scope(self, scope):
        if not scope:
            return ""
        allowed = set(self.scope.split())
        scopes = scope_to_list(scope)
        return list_to_scope([s for s in scopes if s in allowed])

    def check_redirect_uri(self, redirect_uri):
        return redirect_uri in self.redirect_uris

    def check_client_secret(self, client_secret):
        return secrets.compare_digest(self.client_secret, client_secret)

    def check_endpoint_auth_method(self, method, endpoint):
        if endpoint == "token":
            return self.token_endpoint_auth_method == method
        # TODO
        return True

    def check_response_type(self, response_type):
        return response_type in self.response_types

    def check_grant_type(self, grant_type):
        return grant_type in self.grant_types
