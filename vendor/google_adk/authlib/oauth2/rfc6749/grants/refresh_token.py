"""authlib.oauth2.rfc6749.grants.refresh_token.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A special grant endpoint for refresh_token grant_type. Refreshing an
Access Token per `Section 6`_.

.. _`Section 6`: https://tools.ietf.org/html/rfc6749#section-6
"""

import logging

from ..errors import InvalidGrantError
from ..errors import InvalidRequestError
from ..errors import InvalidScopeError
from ..errors import UnauthorizedClientError
from ..hooks import hooked
from ..util import scope_to_list
from .base import BaseGrant
from .base import TokenEndpointMixin

log = logging.getLogger(__name__)


class RefreshTokenGrant(BaseGrant, TokenEndpointMixin):
    """A special grant endpoint for refresh_token grant_type. Refreshing an
    Access Token per `Section 6`_.

    .. _`Section 6`: https://tools.ietf.org/html/rfc6749#section-6
    """

    GRANT_TYPE = "refresh_token"

    #: The authorization server MAY issue a new refresh token
    INCLUDE_NEW_REFRESH_TOKEN = False

    def _validate_request_client(self):
        # require client authentication for confidential clients or for any
        # client that was issued client credentials (or with other
        # authentication requirements)
        client = self.authenticate_token_endpoint_client()
        log.debug("Validate token request of %r", client)

        if not client.check_grant_type(self.GRANT_TYPE):
            raise UnauthorizedClientError(
                f"The client is not authorized to use 'grant_type={self.GRANT_TYPE}'"
            )

        return client

    def _validate_request_token(self, client):
        refresh_token = self.request.form.get("refresh_token")
        if refresh_token is None:
            raise InvalidRequestError("Missing 'refresh_token' in request.")

        token = self.authenticate_refresh_token(refresh_token)
        if not token or not token.check_client(client):
            raise InvalidGrantError()
        return token

    def _validate_token_scope(self, token):
        scope = self.request.payload.scope
        if not scope:
            return

        original_scope = token.get_scope()
        if not original_scope:
            raise InvalidScopeError()

        original_scope = set(scope_to_list(original_scope))
        if not original_scope.issuperset(set(scope_to_list(scope))):
            raise InvalidScopeError()

    def validate_token_request(self):
        """If the authorization server issued a refresh token to the client, the
        client makes a refresh request to the token endpoint by adding the
        following parameters using the "application/x-www-form-urlencoded"
        format per Appendix B with a character encoding of UTF-8 in the HTTP
        request entity-body, per Section 6:

        grant_type
             REQUIRED.  Value MUST be set to "refresh_token".

        refresh_token
             REQUIRED.  The refresh token issued to the client.

        scope
             OPTIONAL.  The scope of the access request as described by
             Section 3.3.  The requested scope MUST NOT include any scope
             not originally granted by the resource owner, and if omitted is
             treated as equal to the scope originally granted by the
             resource owner.


        For example, the client makes the following HTTP request using
        transport-layer security (with extra line breaks for display purposes
        only):

        .. code-block:: http

            POST /token HTTP/1.1
            Host: server.example.com
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
            Content-Type: application/x-www-form-urlencoded

            grant_type=refresh_token&refresh_token=tGzv3JOkF0XG5Qx2TlKWIA
        """
        client = self._validate_request_client()
        self.request.client = client
        refresh_token = self._validate_request_token(client)
        self._validate_token_scope(refresh_token)
        self.request.refresh_token = refresh_token

    @hooked
    def create_token_response(self):
        """If valid and authorized, the authorization server issues an access
        token as described in Section 5.1.  If the request failed
        verification or is invalid, the authorization server returns an error
        response as described in Section 5.2.
        """
        refresh_token = self.request.refresh_token
        user = self.authenticate_user(refresh_token)
        if not user:
            raise InvalidRequestError("There is no 'user' for this token.")

        client = self.request.client
        token = self.issue_token(user, refresh_token)
        log.debug("Issue token %r to %r", token, client)

        self.request.user = user
        self.save_token(token)
        self.revoke_old_credential(refresh_token)
        return 200, token, self.TOKEN_RESPONSE_HEADER

    def issue_token(self, user, refresh_token):
        scope = self.request.payload.scope
        if not scope:
            scope = refresh_token.get_scope()

        token = self.generate_token(
            user=user,
            scope=scope,
            include_refresh_token=self.INCLUDE_NEW_REFRESH_TOKEN,
        )
        return token

    def authenticate_refresh_token(self, refresh_token):
        """Get token information with refresh_token string. Developers MUST
        implement this method in subclass::

            def authenticate_refresh_token(self, refresh_token):
                token = Token.get(refresh_token=refresh_token)
                if token and not token.refresh_token_revoked:
                    return token

        :param refresh_token: The refresh token issued to the client
        :return: token
        """
        raise NotImplementedError()

    def authenticate_user(self, refresh_token):
        """Authenticate the user related to this credential. Developers MUST
        implement this method in subclass::

            def authenticate_user(self, credential):
                return User.get(credential.user_id)

        :param refresh_token: Token object
        :return: user
        """
        raise NotImplementedError()

    def revoke_old_credential(self, refresh_token):
        """The authorization server MAY revoke the old refresh token after
        issuing a new refresh token to the client. Developers MUST implement
        this method in subclass::

            def revoke_old_credential(self, refresh_token):
                credential.revoked = True
                credential.save()

        :param refresh_token: Token object
        """
        raise NotImplementedError()
