from authlib.common.urls import add_params_to_uri
from authlib.common.urls import is_valid_url

from .base_server import BaseServer
from .errors import AccessDeniedError
from .errors import InvalidClientError
from .errors import InvalidRequestError
from .errors import InvalidTokenError
from .errors import MethodNotAllowedError
from .errors import MissingRequiredParameterError
from .errors import OAuth1Error


class AuthorizationServer(BaseServer):
    TOKEN_RESPONSE_HEADER = [
        ("Content-Type", "application/x-www-form-urlencoded"),
        ("Cache-Control", "no-store"),
        ("Pragma", "no-cache"),
    ]

    TEMPORARY_CREDENTIALS_METHOD = "POST"

    def _get_client(self, request):
        client = self.get_client_by_id(request.client_id)
        request.client = client
        return client

    def create_oauth1_request(self, request):
        raise NotImplementedError()

    def handle_response(self, status_code, payload, headers):
        raise NotImplementedError()

    def handle_error_response(self, error):
        return self.handle_response(
            error.status_code, error.get_body(), error.get_headers()
        )

    def validate_temporary_credentials_request(self, request):
        """Validate HTTP request for temporary credentials."""
        # The client obtains a set of temporary credentials from the server by
        # making an authenticated (Section 3) HTTP "POST" request to the
        # Temporary Credential Request endpoint (unless the server advertises
        # another HTTP request method for the client to use).
        if request.method.upper() != self.TEMPORARY_CREDENTIALS_METHOD:
            raise MethodNotAllowedError()

        # REQUIRED parameter
        if not request.client_id:
            raise MissingRequiredParameterError("oauth_consumer_key")

        # REQUIRED parameter
        oauth_callback = request.redirect_uri
        if not request.redirect_uri:
            raise MissingRequiredParameterError("oauth_callback")

        # An absolute URI or
        # other means (the parameter value MUST be set to "oob"
        if oauth_callback != "oob" and not is_valid_url(oauth_callback):
            raise InvalidRequestError('Invalid "oauth_callback" value')

        client = self._get_client(request)
        if not client:
            raise InvalidClientError()

        self.validate_timestamp_and_nonce(request)
        self.validate_oauth_signature(request)
        return request

    def create_temporary_credentials_response(self, request=None):
        """Validate temporary credentials token request and create response
        for temporary credentials token. Assume the endpoint of temporary
        credentials request is ``https://photos.example.net/initiate``:

        .. code-block:: http

            POST /initiate HTTP/1.1
            Host: photos.example.net
            Authorization: OAuth realm="Photos",
                oauth_consumer_key="dpf43f3p2l4k3l03",
                oauth_signature_method="HMAC-SHA1",
                oauth_timestamp="137131200",
                oauth_nonce="wIjqoS",
                oauth_callback="http%3A%2F%2Fprinter.example.com%2Fready",
                oauth_signature="74KNZJeDHnMBp0EMJ9ZHt%2FXKycU%3D"

        The server validates the request and replies with a set of temporary
        credentials in the body of the HTTP response:

        .. code-block:: http

            HTTP/1.1 200 OK
            Content-Type: application/x-www-form-urlencoded

            oauth_token=hh5s93j4hdidpola&oauth_token_secret=hdhd0244k9j7ao03&
            oauth_callback_confirmed=true

        :param request: OAuth1Request instance.
        :returns: (status_code, body, headers)
        """
        try:
            request = self.create_oauth1_request(request)
            self.validate_temporary_credentials_request(request)
        except OAuth1Error as error:
            return self.handle_error_response(error)

        credential = self.create_temporary_credential(request)
        payload = [
            ("oauth_token", credential.get_oauth_token()),
            ("oauth_token_secret", credential.get_oauth_token_secret()),
            ("oauth_callback_confirmed", True),
        ]
        return self.handle_response(200, payload, self.TOKEN_RESPONSE_HEADER)

    def validate_authorization_request(self, request):
        """Validate the request for resource owner authorization."""
        if not request.token:
            raise MissingRequiredParameterError("oauth_token")

        credential = self.get_temporary_credential(request)
        if not credential:
            raise InvalidTokenError()

        # assign credential for later use
        request.credential = credential
        return request

    def create_authorization_response(self, request, grant_user=None):
        """Validate authorization request and create authorization response.
        Assume the endpoint for authorization request is
        ``https://photos.example.net/authorize``, the client redirects Jane's
        user-agent to the server's Resource Owner Authorization endpoint to
        obtain Jane's approval for accessing her private photos::

            https://photos.example.net/authorize?oauth_token=hh5s93j4hdidpola

        The server requests Jane to sign in using her username and password
        and if successful, asks her to approve granting 'printer.example.com'
        access to her private photos.  Jane approves the request and her
        user-agent is redirected to the callback URI provided by the client
        in the previous request (line breaks are for display purposes only)::

            http://printer.example.com/ready?
            oauth_token=hh5s93j4hdidpola&oauth_verifier=hfdp7dh39dks9884

        :param request: OAuth1Request instance.
        :param grant_user: if granted, pass the grant user, otherwise None.
        :returns: (status_code, body, headers)
        """
        request = self.create_oauth1_request(request)
        # authorize endpoint should try catch this error
        self.validate_authorization_request(request)

        temporary_credentials = request.credential
        redirect_uri = temporary_credentials.get_redirect_uri()
        if not redirect_uri or redirect_uri == "oob":
            client_id = temporary_credentials.get_client_id()
            client = self.get_client_by_id(client_id)
            redirect_uri = client.get_default_redirect_uri()

        if grant_user is None:
            error = AccessDeniedError()
            location = add_params_to_uri(redirect_uri, error.get_body())
            return self.handle_response(302, "", [("Location", location)])

        request.user = grant_user
        verifier = self.create_authorization_verifier(request)

        params = [("oauth_token", request.token), ("oauth_verifier", verifier)]
        location = add_params_to_uri(redirect_uri, params)
        return self.handle_response(302, "", [("Location", location)])

    def validate_token_request(self, request):
        """Validate request for issuing token."""
        if not request.client_id:
            raise MissingRequiredParameterError("oauth_consumer_key")

        client = self._get_client(request)
        if not client:
            raise InvalidClientError()

        if not request.token:
            raise MissingRequiredParameterError("oauth_token")

        token = self.get_temporary_credential(request)
        if not token:
            raise InvalidTokenError()

        verifier = request.oauth_params.get("oauth_verifier")
        if not verifier:
            raise MissingRequiredParameterError("oauth_verifier")

        if not token.check_verifier(verifier):
            raise InvalidRequestError('Invalid "oauth_verifier"')

        request.credential = token
        self.validate_timestamp_and_nonce(request)
        self.validate_oauth_signature(request)
        return request

    def create_token_response(self, request):
        """Validate token request and create token response. Assuming the
        endpoint of token request is ``https://photos.example.net/token``,
        the callback request informs the client that Jane completed the
        authorization process.  The client then requests a set of token
        credentials using its temporary credentials (over a secure Transport
        Layer Security (TLS) channel):

        .. code-block:: http

            POST /token HTTP/1.1
            Host: photos.example.net
            Authorization: OAuth realm="Photos",
                oauth_consumer_key="dpf43f3p2l4k3l03",
                oauth_token="hh5s93j4hdidpola",
                oauth_signature_method="HMAC-SHA1",
                oauth_timestamp="137131201",
                oauth_nonce="walatlh",
                oauth_verifier="hfdp7dh39dks9884",
                oauth_signature="gKgrFCywp7rO0OXSjdot%2FIHF7IU%3D"

        The server validates the request and replies with a set of token
        credentials in the body of the HTTP response:

        .. code-block:: http

            HTTP/1.1 200 OK
            Content-Type: application/x-www-form-urlencoded

            oauth_token=nnch734d00sl2jdk&oauth_token_secret=pfkkdhi9sl3r4s00

        :param request: OAuth1Request instance.
        :returns: (status_code, body, headers)
        """
        try:
            request = self.create_oauth1_request(request)
        except OAuth1Error as error:
            return self.handle_error_response(error)

        try:
            self.validate_token_request(request)
        except OAuth1Error as error:
            self.delete_temporary_credential(request)
            return self.handle_error_response(error)

        credential = self.create_token_credential(request)
        payload = [
            ("oauth_token", credential.get_oauth_token()),
            ("oauth_token_secret", credential.get_oauth_token_secret()),
        ]
        self.delete_temporary_credential(request)
        return self.handle_response(200, payload, self.TOKEN_RESPONSE_HEADER)

    def create_temporary_credential(self, request):
        """Generate and save a temporary credential into database or cache.
        A temporary credential is used for exchanging token credential. This
        method should be re-implemented::

            def create_temporary_credential(self, request):
                oauth_token = generate_token(36)
                oauth_token_secret = generate_token(48)
                temporary_credential = TemporaryCredential(
                    oauth_token=oauth_token,
                    oauth_token_secret=oauth_token_secret,
                    client_id=request.client_id,
                    redirect_uri=request.redirect_uri,
                )
                # if the credential has a save method
                temporary_credential.save()
                return temporary_credential

        :param request: OAuth1Request instance
        :return: TemporaryCredential instance
        """
        raise NotImplementedError()

    def get_temporary_credential(self, request):
        """Get the temporary credential from database or cache. A temporary
        credential should share the same methods as described in models of
        ``TemporaryCredentialMixin``::

            def get_temporary_credential(self, request):
                key = "a-key-prefix:{}".format(request.token)
                data = cache.get(key)
                # TemporaryCredential shares methods from TemporaryCredentialMixin
                return TemporaryCredential(data)

        :param request: OAuth1Request instance
        :return: TemporaryCredential instance
        """
        raise NotImplementedError()

    def delete_temporary_credential(self, request):
        """Delete temporary credential from database or cache. For instance,
        if temporary credential is saved in cache::

            def delete_temporary_credential(self, request):
                key = "a-key-prefix:{}".format(request.token)
                cache.delete(key)

        :param request: OAuth1Request instance
        """
        raise NotImplementedError()

    def create_authorization_verifier(self, request):
        """Create and bind ``oauth_verifier`` to temporary credential. It
        could be re-implemented in this way::

            def create_authorization_verifier(self, request):
                verifier = generate_token(36)

                temporary_credential = request.credential
                user_id = request.user.id

                temporary_credential.user_id = user_id
                temporary_credential.oauth_verifier = verifier
                # if the credential has a save method
                temporary_credential.save()

                # remember to return the verifier
                return verifier

        :param request: OAuth1Request instance
        :return: A string of ``oauth_verifier``
        """
        raise NotImplementedError()

    def create_token_credential(self, request):
        """Create and save token credential into database. This method would
        be re-implemented like this::

            def create_token_credential(self, request):
                oauth_token = generate_token(36)
                oauth_token_secret = generate_token(48)
                temporary_credential = request.credential

                token_credential = TokenCredential(
                    oauth_token=oauth_token,
                    oauth_token_secret=oauth_token_secret,
                    client_id=temporary_credential.get_client_id(),
                    user_id=temporary_credential.get_user_id(),
                )
                # if the credential has a save method
                token_credential.save()
                return token_credential

        :param request: OAuth1Request instance
        :return: TokenCredential instance
        """
        raise NotImplementedError()
