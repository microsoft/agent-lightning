from authlib.oidc.discovery.models import _validate_boolean_value


class AuthorizationServerMetadata(dict):
    REGISTRY_KEYS = ["require_signed_request_object"]

    def validate_require_signed_request_object(self):
        """Indicates where authorization request needs to be protected as Request Object and provided through either request or request_uri parameter."""
        _validate_boolean_value(self, "require_signed_request_object")
