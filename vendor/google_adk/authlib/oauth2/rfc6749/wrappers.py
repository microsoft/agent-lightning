import time


class OAuth2Token(dict):
    def __init__(self, params):
        if params.get("expires_at"):
            params["expires_at"] = int(params["expires_at"])
        elif params.get("expires_in"):
            params["expires_at"] = int(time.time()) + int(params["expires_in"])
        super().__init__(params)

    def is_expired(self, leeway=60):
        expires_at = self.get("expires_at")
        if not expires_at:
            return None
        # small timedelta to consider token as expired before it actually expires
        expiration_threshold = expires_at - leeway
        return expiration_threshold < time.time()

    @classmethod
    def from_dict(cls, token):
        if isinstance(token, dict) and not isinstance(token, cls):
            token = cls(token)
        return token
