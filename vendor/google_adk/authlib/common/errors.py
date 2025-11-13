from authlib.consts import default_json_headers


class AuthlibBaseError(Exception):
    """Base Exception for all errors in Authlib."""

    #: short-string error code
    error = None
    #: long-string to describe this error
    description = ""
    #: web page that describes this error
    uri = None

    def __init__(self, error=None, description=None, uri=None):
        if error is not None:
            self.error = error
        if description is not None:
            self.description = description
        if uri is not None:
            self.uri = uri

        message = f"{self.error}: {self.description}"
        super().__init__(message)

    def __repr__(self):
        return f'<{self.__class__.__name__} "{self.error}">'


class AuthlibHTTPError(AuthlibBaseError):
    #: HTTP status code
    status_code = 400

    def __init__(self, error=None, description=None, uri=None, status_code=None):
        super().__init__(error, description, uri)
        if status_code is not None:
            self.status_code = status_code

    def get_error_description(self):
        return self.description

    def get_body(self):
        error = [("error", self.error)]

        if self.description:
            error.append(("error_description", self.description))

        if self.uri:
            error.append(("error_uri", self.uri))
        return error

    def get_headers(self):
        return default_json_headers[:]

    def __call__(self, uri=None):
        self.uri = uri
        body = dict(self.get_body())
        headers = self.get_headers()
        return self.status_code, body, headers


class ContinueIteration(AuthlibBaseError):
    pass
