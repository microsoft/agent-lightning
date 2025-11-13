import json
import time


class FrameworkIntegration:
    expires_in = 3600

    def __init__(self, name, cache=None):
        self.name = name
        self.cache = cache

    def _get_cache_data(self, key):
        value = self.cache.get(key)
        if not value:
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None

    def _clear_session_state(self, session):
        now = time.time()
        for key in dict(session):
            if "_authlib_" in key:
                # TODO: remove in future
                session.pop(key)
            elif key.startswith("_state_"):
                value = session[key]
                exp = value.get("exp")
                if not exp or exp < now:
                    session.pop(key)

    def get_state_data(self, session, state):
        key = f"_state_{self.name}_{state}"
        if self.cache:
            value = self._get_cache_data(key)
        else:
            value = session.get(key)
        if value:
            return value.get("data")
        return None

    def set_state_data(self, session, state, data):
        key = f"_state_{self.name}_{state}"
        if self.cache:
            self.cache.set(key, json.dumps({"data": data}), self.expires_in)
        else:
            now = time.time()
            session[key] = {"data": data, "exp": now + self.expires_in}

    def clear_state_data(self, session, state):
        key = f"_state_{self.name}_{state}"
        if self.cache:
            self.cache.delete(key)
        else:
            session.pop(key, None)
            self._clear_session_state(session)

    def update_token(self, token, refresh_token=None, access_token=None):
        raise NotImplementedError()

    @staticmethod
    def load_config(oauth, name, params):
        raise NotImplementedError()
