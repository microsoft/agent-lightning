import time

from authlib.common.encoding import to_native
from authlib.common.urls import add_params_to_uri
from authlib.common.urls import quote_url
from authlib.jose import jwt
from authlib.oauth2.rfc6749 import InvalidRequestError
from authlib.oauth2.rfc6749 import scope_to_list

from ..errors import AccountSelectionRequiredError
from ..errors import ConsentRequiredError
from ..errors import LoginRequiredError
from ..util import create_half_hash


def is_openid_scope(scope):
    scopes = scope_to_list(scope)
    return scopes and "openid" in scopes


def validate_request_prompt(grant, redirect_uri, redirect_fragment=False):
    prompt = grant.request.payload.data.get("prompt")
    end_user = grant.request.user
    if not prompt:
        if not end_user:
            grant.prompt = "login"
        return grant

    if prompt == "none" and not end_user:
        raise LoginRequiredError(
            redirect_uri=redirect_uri, redirect_fragment=redirect_fragment
        )

    prompts = prompt.split()
    if "none" in prompts and len(prompts) > 1:
        # If this parameter contains none with any other value,
        # an error is returned
        raise InvalidRequestError(
            "Invalid 'prompt' parameter.",
            redirect_uri=redirect_uri,
            redirect_fragment=redirect_fragment,
        )

    prompt = _guess_prompt_value(
        end_user, prompts, redirect_uri, redirect_fragment=redirect_fragment
    )
    if prompt:
        grant.prompt = prompt
    return grant


def validate_nonce(request, exists_nonce, required=False):
    nonce = request.payload.data.get("nonce")
    if not nonce:
        if required:
            raise InvalidRequestError("Missing 'nonce' in request.")
        return True

    if exists_nonce(nonce, request):
        raise InvalidRequestError("Replay attack")


def generate_id_token(
    token,
    user_info,
    key,
    iss,
    aud,
    alg="RS256",
    exp=3600,
    nonce=None,
    auth_time=None,
    acr=None,
    amr=None,
    code=None,
    kid=None,
):
    now = int(time.time())
    if auth_time is None:
        auth_time = now

    header = {"alg": alg}
    if kid:
        header["kid"] = kid

    payload = {
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + exp,
        "auth_time": auth_time,
    }
    if nonce:
        payload["nonce"] = nonce

    if acr:
        payload["acr"] = acr

    if amr:
        payload["amr"] = amr

    if code:
        c_hash = create_half_hash(code, alg)
        if c_hash is not None:
            payload["c_hash"] = to_native(c_hash)

    access_token = token.get("access_token")
    if access_token:
        at_hash = create_half_hash(access_token, alg)
        if at_hash is not None:
            payload["at_hash"] = to_native(at_hash)

    payload.update(user_info)
    return to_native(jwt.encode(header, payload, key))


def create_response_mode_response(redirect_uri, params, response_mode):
    if response_mode == "form_post":
        tpl = (
            "<html><head><title>Redirecting</title></head>"
            '<body onload="javascript:document.forms[0].submit()">'
            '<form method="post" action="{}">{}</form></body></html>'
        )
        inputs = "".join(
            [
                f'<input type="hidden" name="{quote_url(k)}" value="{quote_url(v)}"/>'
                for k, v in params
            ]
        )
        body = tpl.format(quote_url(redirect_uri), inputs)
        return 200, body, [("Content-Type", "text/html; charset=utf-8")]

    if response_mode == "query":
        uri = add_params_to_uri(redirect_uri, params, fragment=False)
    elif response_mode == "fragment":
        uri = add_params_to_uri(redirect_uri, params, fragment=True)
    else:
        raise InvalidRequestError('Invalid "response_mode" value')

    return 302, "", [("Location", uri)]


def _guess_prompt_value(end_user, prompts, redirect_uri, redirect_fragment):
    # http://openid.net/specs/openid-connect-core-1_0.html#AuthRequest

    if not end_user or "login" in prompts:
        return "login"

    if "consent" in prompts:
        if not end_user:
            raise ConsentRequiredError(
                redirect_uri=redirect_uri, redirect_fragment=redirect_fragment
            )
        return "consent"
    elif "select_account" in prompts:
        if not end_user:
            raise AccountSelectionRequiredError(
                redirect_uri=redirect_uri, redirect_fragment=redirect_fragment
            )
        return "select_account"
