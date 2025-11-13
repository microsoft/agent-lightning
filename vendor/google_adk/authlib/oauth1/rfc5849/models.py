class ClientMixin:
    def get_default_redirect_uri(self):
        """A method to get client default redirect_uri. For instance, the
        database table for client has a column called ``default_redirect_uri``::

            def get_default_redirect_uri(self):
                return self.default_redirect_uri

        :return: A URL string
        """
        raise NotImplementedError()

    def get_client_secret(self):
        """A method to return the client_secret of this client. For instance,
        the database table has a column called ``client_secret``::

            def get_client_secret(self):
                return self.client_secret
        """
        raise NotImplementedError()

    def get_rsa_public_key(self):
        """A method to get the RSA public key for RSA-SHA1 signature method.
        For instance, the value is saved on column ``rsa_public_key``::

            def get_rsa_public_key(self):
                return self.rsa_public_key
        """
        raise NotImplementedError()


class TokenCredentialMixin:
    def get_oauth_token(self):
        """A method to get the value of ``oauth_token``. For instance, the
        database table has a column called ``oauth_token``::

            def get_oauth_token(self):
                return self.oauth_token

        :return: A string
        """
        raise NotImplementedError()

    def get_oauth_token_secret(self):
        """A method to get the value of ``oauth_token_secret``. For instance,
        the database table has a column called ``oauth_token_secret``::

            def get_oauth_token_secret(self):
                return self.oauth_token_secret

        :return: A string
        """
        raise NotImplementedError()


class TemporaryCredentialMixin(TokenCredentialMixin):
    def get_client_id(self):
        """A method to get the client_id associated with this credential.
        For instance, the table in the database has a column ``client_id``::

            def get_client_id(self):
                return self.client_id
        """
        raise NotImplementedError()

    def get_redirect_uri(self):
        """A method to get temporary credential's ``oauth_callback``.
        For instance, the database table for temporary credential has a
        column called ``oauth_callback``::

            def get_redirect_uri(self):
                return self.oauth_callback

        :return: A URL string
        """
        raise NotImplementedError()

    def check_verifier(self, verifier):
        """A method to check if the given verifier matches this temporary
        credential. For instance that this temporary credential has recorded
        the value in database as column ``oauth_verifier``::

            def check_verifier(self, verifier):
                return self.oauth_verifier == verifier

        :return: Boolean
        """
        raise NotImplementedError()


class TemporaryCredential(dict, TemporaryCredentialMixin):
    def get_client_id(self):
        return self.get("client_id")

    def get_user_id(self):
        return self.get("user_id")

    def get_redirect_uri(self):
        return self.get("oauth_callback")

    def check_verifier(self, verifier):
        return self.get("oauth_verifier") == verifier

    def get_oauth_token(self):
        return self.get("oauth_token")

    def get_oauth_token_secret(self):
        return self.get("oauth_token_secret")
