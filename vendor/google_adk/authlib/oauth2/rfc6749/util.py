import base64
import binascii
from urllib.parse import unquote

from authlib.common.encoding import to_unicode


def list_to_scope(scope):
    """Convert a list of scopes to a space separated string."""
    if isinstance(scope, (set, tuple, list)):
        return " ".join([to_unicode(s) for s in scope])
    if scope is None:
        return scope
    return to_unicode(scope)


def scope_to_list(scope):
    """Convert a space separated string to a list of scopes."""
    if isinstance(scope, (tuple, list, set)):
        return [to_unicode(s) for s in scope]
    elif scope is None:
        return None
    return scope.strip().split()


def extract_basic_authorization(headers):
    auth = headers.get("Authorization")
    if not auth or " " not in auth:
        return None, None

    auth_type, auth_token = auth.split(None, 1)
    if auth_type.lower() != "basic":
        return None, None

    try:
        query = to_unicode(base64.b64decode(auth_token))
    except (binascii.Error, TypeError):
        return None, None
    if ":" in query:
        username, password = query.split(":", 1)
        return unquote(username), unquote(password)
    return query, None
