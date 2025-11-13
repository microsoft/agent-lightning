from authlib.common.urls import urlparse


def get_well_known_url(issuer, external=False):
    """Get well-known URI with issuer via Section 4.1.

    :param issuer: URL of the issuer
    :param external: return full external url or not
    :return: URL
    """
    # https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderConfigurationRequest
    if external:
        return issuer.rstrip("/") + "/.well-known/openid-configuration"

    parsed = urlparse.urlparse(issuer)
    path = parsed.path
    return path.rstrip("/") + "/.well-known/openid-configuration"
