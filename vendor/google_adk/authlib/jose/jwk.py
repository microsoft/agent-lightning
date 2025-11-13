from authlib.deprecate import deprecate

from .rfc7517 import JsonWebKey


def loads(obj, kid=None):
    deprecate("Please use ``JsonWebKey`` directly.")
    key_set = JsonWebKey.import_key_set(obj)
    if key_set:
        return key_set.find_by_kid(kid)
    return JsonWebKey.import_key(obj)


def dumps(key, kty=None, **params):
    deprecate("Please use ``JsonWebKey`` directly.")
    if kty:
        params["kty"] = kty

    key = JsonWebKey.import_key(key, params)
    return dict(key)
