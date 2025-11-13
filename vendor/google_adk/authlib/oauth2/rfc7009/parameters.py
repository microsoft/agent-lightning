from authlib.common.urls import add_params_to_qs


def prepare_revoke_token_request(token, token_type_hint=None, body=None, headers=None):
    """Construct request body and headers for revocation endpoint.

    :param token: access_token or refresh_token string.
    :param token_type_hint: Optional, `access_token` or `refresh_token`.
    :param body: current request body.
    :param headers: current request headers.
    :return: tuple of (body, headers)

    https://tools.ietf.org/html/rfc7009#section-2.1
    """
    params = [("token", token)]
    if token_type_hint:
        params.append(("token_type_hint", token_type_hint))

    body = add_params_to_qs(body or "", params)
    if headers is None:
        headers = {}

    headers["Content-Type"] = "application/x-www-form-urlencoded"
    return body, headers
