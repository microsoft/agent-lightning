from authlib.jose import JsonWebKey
from authlib.jose import JsonWebToken
from authlib.jose import jwt
from authlib.oidc.core import CodeIDToken
from authlib.oidc.core import ImplicitIDToken
from authlib.oidc.core import UserInfo


class OpenIDMixin:
    def fetch_jwk_set(self, force=False):
        metadata = self.load_server_metadata()
        jwk_set = metadata.get("jwks")
        if jwk_set and not force:
            return jwk_set

        uri = metadata.get("jwks_uri")
        if not uri:
            raise RuntimeError('Missing "jwks_uri" in metadata')

        with self.client_cls(**self.client_kwargs) as session:
            resp = session.request("GET", uri, withhold_token=True)
            resp.raise_for_status()
            jwk_set = resp.json()

        self.server_metadata["jwks"] = jwk_set
        return jwk_set

    def userinfo(self, **kwargs):
        """Fetch user info from ``userinfo_endpoint``."""
        metadata = self.load_server_metadata()
        resp = self.get(metadata["userinfo_endpoint"], **kwargs)
        resp.raise_for_status()
        data = resp.json()
        return UserInfo(data)

    def parse_id_token(
        self, token, nonce, claims_options=None, claims_cls=None, leeway=120
    ):
        """Return an instance of UserInfo from token's ``id_token``."""
        if "id_token" not in token:
            return None

        load_key = self.create_load_key()

        claims_params = dict(
            nonce=nonce,
            client_id=self.client_id,
        )

        if claims_cls is None:
            if "access_token" in token:
                claims_params["access_token"] = token["access_token"]
                claims_cls = CodeIDToken
            else:
                claims_cls = ImplicitIDToken

        metadata = self.load_server_metadata()
        if claims_options is None and "issuer" in metadata:
            claims_options = {"iss": {"values": [metadata["issuer"]]}}

        alg_values = metadata.get("id_token_signing_alg_values_supported")
        if alg_values:
            _jwt = JsonWebToken(alg_values)
        else:
            _jwt = jwt

        claims = _jwt.decode(
            token["id_token"],
            key=load_key,
            claims_cls=claims_cls,
            claims_options=claims_options,
            claims_params=claims_params,
        )
        # https://github.com/authlib/authlib/issues/259
        if claims.get("nonce_supported") is False:
            claims.params["nonce"] = None

        claims.validate(leeway=leeway)
        return UserInfo(claims)

    def create_load_key(self):
        def load_key(header, _):
            jwk_set = JsonWebKey.import_key_set(self.fetch_jwk_set())
            try:
                return jwk_set.find_by_kid(
                    header.get("kid"), use="sig", alg=header.get("alg")
                )
            except ValueError:
                # re-try with new jwk set
                jwk_set = JsonWebKey.import_key_set(self.fetch_jwk_set(force=True))
                return jwk_set.find_by_kid(
                    header.get("kid"), use="sig", alg=header.get("alg")
                )

        return load_key
