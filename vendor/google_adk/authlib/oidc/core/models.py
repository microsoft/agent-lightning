from authlib.oauth2.rfc6749 import AuthorizationCodeMixin as _AuthorizationCodeMixin


class AuthorizationCodeMixin(_AuthorizationCodeMixin):
    def get_nonce(self):
        """Get "nonce" value of the authorization code object."""
        # OPs MUST support the prompt parameter, as defined in Section 3.1.2, including the specified user interface behaviors such as none and login.
        raise NotImplementedError()

    def get_auth_time(self):
        """Get "auth_time" value of the authorization code object."""
        # OPs MUST support returning the time at which the End-User authenticated via the auth_time Claim, when requested, as defined in Section 2.
        raise NotImplementedError()

    def get_acr(self) -> str:
        """Get the "acr" (Authentication Method Class) value of the authorization code object."""
        # OPs MUST support requests for specific Authentication Context Class Reference values via the acr_values parameter, as defined in Section 3.1.2. (Note that the minimum level of support required for this parameter is simply to have its use not result in an error.)
        return None

    def get_amr(self) -> list[str]:
        """Get the "amr" (Authentication Method Reference) value of the authorization code object.

        Have a look at :rfc:`RFC8176 <8176>` to see the full list of registered amr.

            def get_amr(self) -> list[str]:
                return ["pwd", "otp"]

        """
        return None
