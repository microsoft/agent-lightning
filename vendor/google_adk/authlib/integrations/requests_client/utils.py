REQUESTS_SESSION_KWARGS = [
    "proxies",
    "hooks",
    "stream",
    "verify",
    "cert",
    "max_redirects",
    "trust_env",
]


def update_session_configure(session, kwargs):
    for k in REQUESTS_SESSION_KWARGS:
        if k in kwargs:
            setattr(session, k, kwargs.pop(k))
