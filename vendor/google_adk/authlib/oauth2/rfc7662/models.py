from ..rfc6749 import TokenMixin


class IntrospectionToken(dict, TokenMixin):
    def get_client_id(self):
        return self.get("client_id")

    def get_scope(self):
        return self.get("scope")

    def get_expires_in(self):
        # this method is only used in refresh token,
        # no need to implement it
        return 0

    def get_expires_at(self):
        return self.get("exp", 0)

    def __getattr__(self, key):
        # https://tools.ietf.org/html/rfc7662#section-2.2
        available_keys = {
            "active",
            "scope",
            "client_id",
            "username",
            "token_type",
            "exp",
            "iat",
            "nbf",
            "sub",
            "aud",
            "iss",
            "jti",
        }
        try:
            return object.__getattribute__(self, key)
        except AttributeError as error:
            if key in available_keys:
                return self.get(key)
            raise error
