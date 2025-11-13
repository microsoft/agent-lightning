from authlib.common.errors import AuthlibHTTPError
from authlib.common.urls import add_params_to_uri


def invalid_error_characters(text: str) -> list[str]:
    """Check whether the string only contains characters from the restricted ASCII set defined in RFC6749 for errors.

    https://datatracker.ietf.org/doc/html/rfc6749#section-4.1.2.1
    """
    valid_ranges = [
        (0x20, 0x21),
        (0x23, 0x5B),
        (0x5D, 0x7E),
    ]

    return [
        char
        for char in set(text)
        if not any(start <= ord(char) <= end for start, end in valid_ranges)
    ]


class OAuth2Error(AuthlibHTTPError):
    def __init__(
        self,
        description=None,
        uri=None,
        status_code=None,
        state=None,
        redirect_uri=None,
        redirect_fragment=False,
        error=None,
    ):
        # Human-readable ASCII [USASCII] text providing
        # additional information, used to assist the client developer in
        # understanding the error that occurred.
        # Values for the "error_description" parameter MUST NOT include
        # characters outside the set %x20-21 / %x23-5B / %x5D-7E.
        if description:
            if chars := invalid_error_characters(description):
                raise ValueError(
                    f"Error description contains forbidden characters: {', '.join(chars)}."
                )

        super().__init__(error, description, uri, status_code)
        self.state = state
        self.redirect_uri = redirect_uri
        self.redirect_fragment = redirect_fragment

    def get_body(self):
        """Get a list of body."""
        error = super().get_body()
        if self.state:
            error.append(("state", self.state))
        return error

    def __call__(self, uri=None):
        if self.redirect_uri:
            params = self.get_body()
            loc = add_params_to_uri(self.redirect_uri, params, self.redirect_fragment)
            return 302, "", [("Location", loc)]
        return super().__call__(uri=uri)
