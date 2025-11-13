from authlib.common.encoding import json_dumps


class KeySet:
    """This class represents a JSON Web Key Set."""

    def __init__(self, keys):
        self.keys = keys

    def as_dict(self, is_private=False, **params):
        """Represent this key as a dict of the JSON Web Key Set."""
        return {"keys": [k.as_dict(is_private, **params) for k in self.keys]}

    def as_json(self, is_private=False, **params):
        """Represent this key set as a JSON string."""
        obj = self.as_dict(is_private, **params)
        return json_dumps(obj)

    def find_by_kid(self, kid, **params):
        """Find the key matches the given kid value.

        :param kid: A string of kid
        :return: Key instance
        :raise: ValueError
        """
        # Proposed fix, feel free to do something else but the idea is that we take the only key
        # of the set if no kid is specified
        if kid is None and len(self.keys) == 1:
            return self.keys[0]

        keys = [key for key in self.keys if key.kid == kid]
        if params:
            keys = list(_filter_keys_by_params(keys, **params))

        if keys:
            return keys[0]
        raise ValueError("Key not found")


def _filter_keys_by_params(keys, **params):
    _use = params.get("use")
    _alg = params.get("alg")

    for key in keys:
        designed_use = key.tokens.get("use")
        if designed_use and _use and designed_use != _use:
            continue

        designed_alg = key.tokens.get("alg")
        if designed_alg and _alg and designed_alg != _alg:
            continue

        yield key
