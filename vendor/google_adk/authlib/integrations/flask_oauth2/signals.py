from flask.signals import Namespace

_signal = Namespace()

#: signal when client is authenticated
client_authenticated = _signal.signal("client_authenticated")

#: signal when token is revoked
token_revoked = _signal.signal("token_revoked")

#: signal when token is authenticated
token_authenticated = _signal.signal("token_authenticated")
