# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections import OrderedDict
from http import HTTPStatus
import json
import logging as std_logging
import os
import re
from typing import (
    Dict,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
import uuid
import warnings

from google.cloud.spanner_admin_instance_v1 import gapic_version as package_version

from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)

from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.spanner_admin_instance_v1.services.instance_admin import pagers
from google.cloud.spanner_admin_instance_v1.types import spanner_instance_admin
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from .transports.base import InstanceAdminTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import InstanceAdminGrpcTransport
from .transports.grpc_asyncio import InstanceAdminGrpcAsyncIOTransport
from .transports.rest import InstanceAdminRestTransport


class InstanceAdminClientMeta(type):
    """Metaclass for the InstanceAdmin client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """

    _transport_registry = OrderedDict()  # type: Dict[str, Type[InstanceAdminTransport]]
    _transport_registry["grpc"] = InstanceAdminGrpcTransport
    _transport_registry["grpc_asyncio"] = InstanceAdminGrpcAsyncIOTransport
    _transport_registry["rest"] = InstanceAdminRestTransport

    def get_transport_class(
        cls,
        label: Optional[str] = None,
    ) -> Type[InstanceAdminTransport]:
        """Returns an appropriate transport class.

        Args:
            label: The name of the desired transport. If none is
                provided, then the first transport in the registry is used.

        Returns:
            The transport class to use.
        """
        # If a specific transport is requested, return that one.
        if label:
            return cls._transport_registry[label]

        # No transport is requested; return the default (that is, the first one
        # in the dictionary).
        return next(iter(cls._transport_registry.values()))


class InstanceAdminClient(metaclass=InstanceAdminClientMeta):
    """Cloud Spanner Instance Admin API

    The Cloud Spanner Instance Admin API can be used to create,
    delete, modify and list instances. Instances are dedicated Cloud
    Spanner serving and storage resources to be used by Cloud
    Spanner databases.

    Each instance has a "configuration", which dictates where the
    serving resources for the Cloud Spanner instance are located
    (e.g., US-central, Europe). Configurations are created by Google
    based on resource availability.

    Cloud Spanner billing is based on the instances that exist and
    their sizes. After an instance exists, there are no additional
    per-database or per-operation charges for use of the instance
    (though there may be additional network bandwidth charges).
    Instances offer isolation: problems with databases in one
    instance will not affect other instances. However, within an
    instance databases can affect each other. For example, if one
    database in an instance receives a lot of requests and consumes
    most of the instance resources, fewer resources are available
    for other databases in that instance, and their performance may
    suffer.
    """

    @staticmethod
    def _get_default_mtls_endpoint(api_endpoint):
        """Converts api endpoint to mTLS endpoint.

        Convert "*.sandbox.googleapis.com" and "*.googleapis.com" to
        "*.mtls.sandbox.googleapis.com" and "*.mtls.googleapis.com" respectively.
        Args:
            api_endpoint (Optional[str]): the api endpoint to convert.
        Returns:
            str: converted mTLS api endpoint.
        """
        if not api_endpoint:
            return api_endpoint

        mtls_endpoint_re = re.compile(
            r"(?P<name>[^.]+)(?P<mtls>\.mtls)?(?P<sandbox>\.sandbox)?(?P<googledomain>\.googleapis\.com)?"
        )

        m = mtls_endpoint_re.match(api_endpoint)
        name, mtls, sandbox, googledomain = m.groups()
        if mtls or not googledomain:
            return api_endpoint

        if sandbox:
            return api_endpoint.replace(
                "sandbox.googleapis.com", "mtls.sandbox.googleapis.com"
            )

        return api_endpoint.replace(".googleapis.com", ".mtls.googleapis.com")

    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = "spanner.googleapis.com"
    DEFAULT_MTLS_ENDPOINT = _get_default_mtls_endpoint.__func__(  # type: ignore
        DEFAULT_ENDPOINT
    )

    _DEFAULT_ENDPOINT_TEMPLATE = "spanner.{UNIVERSE_DOMAIN}"
    _DEFAULT_UNIVERSE = "googleapis.com"

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            InstanceAdminClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_info(info)
        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename: str, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            InstanceAdminClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_file(filename)
        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

    from_service_account_json = from_service_account_file

    @property
    def transport(self) -> InstanceAdminTransport:
        """Returns the transport used by the client instance.

        Returns:
            InstanceAdminTransport: The transport used by the client
                instance.
        """
        return self._transport

    @staticmethod
    def instance_path(
        project: str,
        instance: str,
    ) -> str:
        """Returns a fully-qualified instance string."""
        return "projects/{project}/instances/{instance}".format(
            project=project,
            instance=instance,
        )

    @staticmethod
    def parse_instance_path(path: str) -> Dict[str, str]:
        """Parses a instance path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/instances/(?P<instance>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def instance_config_path(
        project: str,
        instance_config: str,
    ) -> str:
        """Returns a fully-qualified instance_config string."""
        return "projects/{project}/instanceConfigs/{instance_config}".format(
            project=project,
            instance_config=instance_config,
        )

    @staticmethod
    def parse_instance_config_path(path: str) -> Dict[str, str]:
        """Parses a instance_config path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/instanceConfigs/(?P<instance_config>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def instance_partition_path(
        project: str,
        instance: str,
        instance_partition: str,
    ) -> str:
        """Returns a fully-qualified instance_partition string."""
        return "projects/{project}/instances/{instance}/instancePartitions/{instance_partition}".format(
            project=project,
            instance=instance,
            instance_partition=instance_partition,
        )

    @staticmethod
    def parse_instance_partition_path(path: str) -> Dict[str, str]:
        """Parses a instance_partition path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/instances/(?P<instance>.+?)/instancePartitions/(?P<instance_partition>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def common_billing_account_path(
        billing_account: str,
    ) -> str:
        """Returns a fully-qualified billing_account string."""
        return "billingAccounts/{billing_account}".format(
            billing_account=billing_account,
        )

    @staticmethod
    def parse_common_billing_account_path(path: str) -> Dict[str, str]:
        """Parse a billing_account path into its component segments."""
        m = re.match(r"^billingAccounts/(?P<billing_account>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_folder_path(
        folder: str,
    ) -> str:
        """Returns a fully-qualified folder string."""
        return "folders/{folder}".format(
            folder=folder,
        )

    @staticmethod
    def parse_common_folder_path(path: str) -> Dict[str, str]:
        """Parse a folder path into its component segments."""
        m = re.match(r"^folders/(?P<folder>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_organization_path(
        organization: str,
    ) -> str:
        """Returns a fully-qualified organization string."""
        return "organizations/{organization}".format(
            organization=organization,
        )

    @staticmethod
    def parse_common_organization_path(path: str) -> Dict[str, str]:
        """Parse a organization path into its component segments."""
        m = re.match(r"^organizations/(?P<organization>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_project_path(
        project: str,
    ) -> str:
        """Returns a fully-qualified project string."""
        return "projects/{project}".format(
            project=project,
        )

    @staticmethod
    def parse_common_project_path(path: str) -> Dict[str, str]:
        """Parse a project path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_location_path(
        project: str,
        location: str,
    ) -> str:
        """Returns a fully-qualified location string."""
        return "projects/{project}/locations/{location}".format(
            project=project,
            location=location,
        )

    @staticmethod
    def parse_common_location_path(path: str) -> Dict[str, str]:
        """Parse a location path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/locations/(?P<location>.+?)$", path)
        return m.groupdict() if m else {}

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[client_options_lib.ClientOptions] = None
    ):
        """Deprecated. Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """

        warnings.warn(
            "get_mtls_endpoint_and_cert_source is deprecated. Use the api_endpoint property instead.",
            DeprecationWarning,
        )
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        use_client_cert = os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`"
            )

        # Figure out the client cert source to use.
        client_cert_source = None
        if use_client_cert == "true":
            if client_options.client_cert_source:
                client_cert_source = client_options.client_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()

        # Figure out which api endpoint to use.
        if client_options.api_endpoint is not None:
            api_endpoint = client_options.api_endpoint
        elif use_mtls_endpoint == "always" or (
            use_mtls_endpoint == "auto" and client_cert_source
        ):
            api_endpoint = cls.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = cls.DEFAULT_ENDPOINT

        return api_endpoint, client_cert_source

    @staticmethod
    def _read_environment_variables():
        """Returns the environment variables used by the client.

        Returns:
            Tuple[bool, str, str]: returns the GOOGLE_API_USE_CLIENT_CERTIFICATE,
            GOOGLE_API_USE_MTLS_ENDPOINT, and GOOGLE_CLOUD_UNIVERSE_DOMAIN environment variables.

        Raises:
            ValueError: If GOOGLE_API_USE_CLIENT_CERTIFICATE is not
                any of ["true", "false"].
            google.auth.exceptions.MutualTLSChannelError: If GOOGLE_API_USE_MTLS_ENDPOINT
                is not any of ["auto", "never", "always"].
        """
        use_client_cert = os.getenv(
            "GOOGLE_API_USE_CLIENT_CERTIFICATE", "false"
        ).lower()
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto").lower()
        universe_domain_env = os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`"
            )
        return use_client_cert == "true", use_mtls_endpoint, universe_domain_env

    @staticmethod
    def _get_client_cert_source(provided_cert_source, use_cert_flag):
        """Return the client cert source to be used by the client.

        Args:
            provided_cert_source (bytes): The client certificate source provided.
            use_cert_flag (bool): A flag indicating whether to use the client certificate.

        Returns:
            bytes or None: The client cert source to be used by the client.
        """
        client_cert_source = None
        if use_cert_flag:
            if provided_cert_source:
                client_cert_source = provided_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()
        return client_cert_source

    @staticmethod
    def _get_api_endpoint(
        api_override, client_cert_source, universe_domain, use_mtls_endpoint
    ):
        """Return the API endpoint used by the client.

        Args:
            api_override (str): The API endpoint override. If specified, this is always
                the return value of this function and the other arguments are not used.
            client_cert_source (bytes): The client certificate source used by the client.
            universe_domain (str): The universe domain used by the client.
            use_mtls_endpoint (str): How to use the mTLS endpoint, which depends also on the other parameters.
                Possible values are "always", "auto", or "never".

        Returns:
            str: The API endpoint to be used by the client.
        """
        if api_override is not None:
            api_endpoint = api_override
        elif use_mtls_endpoint == "always" or (
            use_mtls_endpoint == "auto" and client_cert_source
        ):
            _default_universe = InstanceAdminClient._DEFAULT_UNIVERSE
            if universe_domain != _default_universe:
                raise MutualTLSChannelError(
                    f"mTLS is not supported in any universe other than {_default_universe}."
                )
            api_endpoint = InstanceAdminClient.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = InstanceAdminClient._DEFAULT_ENDPOINT_TEMPLATE.format(
                UNIVERSE_DOMAIN=universe_domain
            )
        return api_endpoint

    @staticmethod
    def _get_universe_domain(
        client_universe_domain: Optional[str], universe_domain_env: Optional[str]
    ) -> str:
        """Return the universe domain used by the client.

        Args:
            client_universe_domain (Optional[str]): The universe domain configured via the client options.
            universe_domain_env (Optional[str]): The universe domain configured via the "GOOGLE_CLOUD_UNIVERSE_DOMAIN" environment variable.

        Returns:
            str: The universe domain to be used by the client.

        Raises:
            ValueError: If the universe domain is an empty string.
        """
        universe_domain = InstanceAdminClient._DEFAULT_UNIVERSE
        if client_universe_domain is not None:
            universe_domain = client_universe_domain
        elif universe_domain_env is not None:
            universe_domain = universe_domain_env
        if len(universe_domain.strip()) == 0:
            raise ValueError("Universe Domain cannot be an empty string.")
        return universe_domain

    def _validate_universe_domain(self):
        """Validates client's and credentials' universe domains are consistent.

        Returns:
            bool: True iff the configured universe domain is valid.

        Raises:
            ValueError: If the configured universe domain is not valid.
        """

        # NOTE (b/349488459): universe validation is disabled until further notice.
        return True

    def _add_cred_info_for_auth_errors(
        self, error: core_exceptions.GoogleAPICallError
    ) -> None:
        """Adds credential info string to error details for 401/403/404 errors.

        Args:
            error (google.api_core.exceptions.GoogleAPICallError): The error to add the cred info.
        """
        if error.code not in [
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.FORBIDDEN,
            HTTPStatus.NOT_FOUND,
        ]:
            return

        cred = self._transport._credentials

        # get_cred_info is only available in google-auth>=2.35.0
        if not hasattr(cred, "get_cred_info"):
            return

        # ignore the type check since pypy test fails when get_cred_info
        # is not available
        cred_info = cred.get_cred_info()  # type: ignore
        if cred_info and hasattr(error._details, "append"):
            error._details.append(json.dumps(cred_info))

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used by the client instance.
        """
        return self._universe_domain

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, InstanceAdminTransport, Callable[..., InstanceAdminTransport]]
        ] = None,
        client_options: Optional[Union[client_options_lib.ClientOptions, dict]] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the instance admin client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,InstanceAdminTransport,Callable[..., InstanceAdminTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the InstanceAdminTransport constructor.
                If set to None, a transport is chosen automatically.
            client_options (Optional[Union[google.api_core.client_options.ClientOptions, dict]]):
                Custom options for the client.

                1. The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client when ``transport`` is
                not explicitly provided. Only if this property is not set and
                ``transport`` was not explicitly provided, the endpoint is
                determined by the GOOGLE_API_USE_MTLS_ENDPOINT environment
                variable, which have one of the following values:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto-switch to the
                default mTLS endpoint if client certificate is present; this is
                the default value).

                2. If the GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide a client certificate for mTLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.

                3. The ``universe_domain`` property can be used to override the
                default "googleapis.com" universe. Note that the ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        self._client_options = client_options
        if isinstance(self._client_options, dict):
            self._client_options = client_options_lib.from_dict(self._client_options)
        if self._client_options is None:
            self._client_options = client_options_lib.ClientOptions()
        self._client_options = cast(
            client_options_lib.ClientOptions, self._client_options
        )

        universe_domain_opt = getattr(self._client_options, "universe_domain", None)

        (
            self._use_client_cert,
            self._use_mtls_endpoint,
            self._universe_domain_env,
        ) = InstanceAdminClient._read_environment_variables()
        self._client_cert_source = InstanceAdminClient._get_client_cert_source(
            self._client_options.client_cert_source, self._use_client_cert
        )
        self._universe_domain = InstanceAdminClient._get_universe_domain(
            universe_domain_opt, self._universe_domain_env
        )
        self._api_endpoint = None  # updated below, depending on `transport`

        # Initialize the universe domain validation.
        self._is_universe_domain_valid = False

        if CLIENT_LOGGING_SUPPORTED:  # pragma: NO COVER
            # Setup logging.
            client_logging.initialize_logging()

        api_key_value = getattr(self._client_options, "api_key", None)
        if api_key_value and credentials:
            raise ValueError(
                "client_options.api_key and credentials are mutually exclusive"
            )

        # Save or instantiate the transport.
        # Ordinarily, we provide the transport, but allowing a custom transport
        # instance provides an extensibility point for unusual situations.
        transport_provided = isinstance(transport, InstanceAdminTransport)
        if transport_provided:
            # transport is a InstanceAdminTransport instance.
            if credentials or self._client_options.credentials_file or api_key_value:
                raise ValueError(
                    "When providing a transport instance, "
                    "provide its credentials directly."
                )
            if self._client_options.scopes:
                raise ValueError(
                    "When providing a transport instance, provide its scopes "
                    "directly."
                )
            self._transport = cast(InstanceAdminTransport, transport)
            self._api_endpoint = self._transport.host

        self._api_endpoint = (
            self._api_endpoint
            or InstanceAdminClient._get_api_endpoint(
                self._client_options.api_endpoint,
                self._client_cert_source,
                self._universe_domain,
                self._use_mtls_endpoint,
            )
        )

        if not transport_provided:
            import google.auth._default  # type: ignore

            if api_key_value and hasattr(
                google.auth._default, "get_api_key_credentials"
            ):
                credentials = google.auth._default.get_api_key_credentials(
                    api_key_value
                )

            transport_init: Union[
                Type[InstanceAdminTransport], Callable[..., InstanceAdminTransport]
            ] = (
                InstanceAdminClient.get_transport_class(transport)
                if isinstance(transport, str) or transport is None
                else cast(Callable[..., InstanceAdminTransport], transport)
            )
            # initialize with the provided callable or the passed in class
            self._transport = transport_init(
                credentials=credentials,
                credentials_file=self._client_options.credentials_file,
                host=self._api_endpoint,
                scopes=self._client_options.scopes,
                client_cert_source_for_mtls=self._client_cert_source,
                quota_project_id=self._client_options.quota_project_id,
                client_info=client_info,
                always_use_jwt_access=True,
                api_audience=self._client_options.api_audience,
            )

        if "async" not in str(self._transport):
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                std_logging.DEBUG
            ):  # pragma: NO COVER
                _LOGGER.debug(
                    "Created client `google.spanner.admin.instance_v1.InstanceAdminClient`.",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "universeDomain": getattr(
                            self._transport._credentials, "universe_domain", ""
                        ),
                        "credentialsType": f"{type(self._transport._credentials).__module__}.{type(self._transport._credentials).__qualname__}",
                        "credentialsInfo": getattr(
                            self.transport._credentials, "get_cred_info", lambda: None
                        )(),
                    }
                    if hasattr(self._transport, "_credentials")
                    else {
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "credentialsType": None,
                    },
                )

    def list_instance_configs(
        self,
        request: Optional[
            Union[spanner_instance_admin.ListInstanceConfigsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListInstanceConfigsPager:
        r"""Lists the supported instance configurations for a
        given project.
        Returns both Google-managed configurations and
        user-managed configurations.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_list_instance_configs():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.ListInstanceConfigsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_instance_configs(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigsRequest, dict]):
                The request object. The request for
                [ListInstanceConfigs][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigs].
            parent (str):
                Required. The name of the project for which a list of
                supported instance configurations is requested. Values
                are of the form ``projects/<project>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.services.instance_admin.pagers.ListInstanceConfigsPager:
                The response for
                   [ListInstanceConfigs][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigs].

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.ListInstanceConfigsRequest):
            request = spanner_instance_admin.ListInstanceConfigsRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_instance_configs]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListInstanceConfigsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_instance_config(
        self,
        request: Optional[
            Union[spanner_instance_admin.GetInstanceConfigRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> spanner_instance_admin.InstanceConfig:
        r"""Gets information about a particular instance
        configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_get_instance_config():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.GetInstanceConfigRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_instance_config(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.GetInstanceConfigRequest, dict]):
                The request object. The request for
                [GetInstanceConfigRequest][google.spanner.admin.instance.v1.InstanceAdmin.GetInstanceConfig].
            name (str):
                Required. The name of the requested instance
                configuration. Values are of the form
                ``projects/<project>/instanceConfigs/<config>``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.types.InstanceConfig:
                A possible configuration for a Cloud
                Spanner instance. Configurations define
                the geographic placement of nodes and
                their replication.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.GetInstanceConfigRequest):
            request = spanner_instance_admin.GetInstanceConfigRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_instance_config]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def create_instance_config(
        self,
        request: Optional[
            Union[spanner_instance_admin.CreateInstanceConfigRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        instance_config: Optional[spanner_instance_admin.InstanceConfig] = None,
        instance_config_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Creates an instance configuration and begins preparing it to be
        used. The returned long-running operation can be used to track
        the progress of preparing the new instance configuration. The
        instance configuration name is assigned by the caller. If the
        named instance configuration already exists,
        ``CreateInstanceConfig`` returns ``ALREADY_EXISTS``.

        Immediately after the request returns:

        - The instance configuration is readable via the API, with all
          requested attributes. The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field is set to true. Its state is ``CREATING``.

        While the operation is pending:

        - Cancelling the operation renders the instance configuration
          immediately unreadable via the API.
        - Except for deleting the creating resource, all other attempts
          to modify the instance configuration are rejected.

        Upon completion of the returned operation:

        - Instances can be created using the instance configuration.
        - The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field becomes false. Its state becomes ``READY``.

        The returned long-running operation will have a name of the
        format ``<instance_config_name>/operations/<operation_id>`` and
        can be used to track creation of the instance configuration. The
        metadata field type is
        [CreateInstanceConfigMetadata][google.spanner.admin.instance.v1.CreateInstanceConfigMetadata].
        The response field type is
        [InstanceConfig][google.spanner.admin.instance.v1.InstanceConfig],
        if successful.

        Authorization requires ``spanner.instanceConfigs.create``
        permission on the resource
        [parent][google.spanner.admin.instance.v1.CreateInstanceConfigRequest.parent].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_create_instance_config():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.CreateInstanceConfigRequest(
                    parent="parent_value",
                    instance_config_id="instance_config_id_value",
                )

                # Make the request
                operation = client.create_instance_config(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.CreateInstanceConfigRequest, dict]):
                The request object. The request for
                [CreateInstanceConfig][google.spanner.admin.instance.v1.InstanceAdmin.CreateInstanceConfig].
            parent (str):
                Required. The name of the project in which to create the
                instance configuration. Values are of the form
                ``projects/<project>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance_config (google.cloud.spanner_admin_instance_v1.types.InstanceConfig):
                Required. The ``InstanceConfig`` proto of the
                configuration to create. ``instance_config.name`` must
                be ``<parent>/instanceConfigs/<instance_config_id>``.
                ``instance_config.base_config`` must be a Google-managed
                configuration name, e.g. /instanceConfigs/us-east1,
                /instanceConfigs/nam3.

                This corresponds to the ``instance_config`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance_config_id (str):
                Required. The ID of the instance configuration to
                create. Valid identifiers are of the form
                ``custom-[-a-z0-9]*[a-z0-9]`` and must be between 2 and
                64 characters in length. The ``custom-`` prefix is
                required to avoid name conflicts with Google-managed
                configurations.

                This corresponds to the ``instance_config_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.spanner_admin_instance_v1.types.InstanceConfig` A possible configuration for a Cloud Spanner instance. Configurations
                   define the geographic placement of nodes and their
                   replication.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, instance_config, instance_config_id]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.CreateInstanceConfigRequest):
            request = spanner_instance_admin.CreateInstanceConfigRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent
            if instance_config is not None:
                request.instance_config = instance_config
            if instance_config_id is not None:
                request.instance_config_id = instance_config_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_instance_config]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.InstanceConfig,
            metadata_type=spanner_instance_admin.CreateInstanceConfigMetadata,
        )

        # Done; return the response.
        return response

    def update_instance_config(
        self,
        request: Optional[
            Union[spanner_instance_admin.UpdateInstanceConfigRequest, dict]
        ] = None,
        *,
        instance_config: Optional[spanner_instance_admin.InstanceConfig] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Updates an instance configuration. The returned long-running
        operation can be used to track the progress of updating the
        instance. If the named instance configuration does not exist,
        returns ``NOT_FOUND``.

        Only user-managed configurations can be updated.

        Immediately after the request returns:

        - The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field is set to true.

        While the operation is pending:

        - Cancelling the operation sets its metadata's
          [cancel_time][google.spanner.admin.instance.v1.UpdateInstanceConfigMetadata.cancel_time].
          The operation is guaranteed to succeed at undoing all changes,
          after which point it terminates with a ``CANCELLED`` status.
        - All other attempts to modify the instance configuration are
          rejected.
        - Reading the instance configuration via the API continues to
          give the pre-request values.

        Upon completion of the returned operation:

        - Creating instances using the instance configuration uses the
          new values.
        - The new values of the instance configuration are readable via
          the API.
        - The instance configuration's
          [reconciling][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
          field becomes false.

        The returned long-running operation will have a name of the
        format ``<instance_config_name>/operations/<operation_id>`` and
        can be used to track the instance configuration modification.
        The metadata field type is
        [UpdateInstanceConfigMetadata][google.spanner.admin.instance.v1.UpdateInstanceConfigMetadata].
        The response field type is
        [InstanceConfig][google.spanner.admin.instance.v1.InstanceConfig],
        if successful.

        Authorization requires ``spanner.instanceConfigs.update``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstanceConfig.name].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_update_instance_config():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.UpdateInstanceConfigRequest(
                )

                # Make the request
                operation = client.update_instance_config(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.UpdateInstanceConfigRequest, dict]):
                The request object. The request for
                [UpdateInstanceConfig][google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstanceConfig].
            instance_config (google.cloud.spanner_admin_instance_v1.types.InstanceConfig):
                Required. The user instance configuration to update,
                which must always include the instance configuration
                name. Otherwise, only fields mentioned in
                [update_mask][google.spanner.admin.instance.v1.UpdateInstanceConfigRequest.update_mask]
                need be included. To prevent conflicts of concurrent
                updates,
                [etag][google.spanner.admin.instance.v1.InstanceConfig.reconciling]
                can be used.

                This corresponds to the ``instance_config`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (google.protobuf.field_mask_pb2.FieldMask):
                Required. A mask specifying which fields in
                [InstanceConfig][google.spanner.admin.instance.v1.InstanceConfig]
                should be updated. The field mask must always be
                specified; this prevents any future fields in
                [InstanceConfig][google.spanner.admin.instance.v1.InstanceConfig]
                from being erased accidentally by clients that do not
                know about them. Only display_name and labels can be
                updated.

                This corresponds to the ``update_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.spanner_admin_instance_v1.types.InstanceConfig` A possible configuration for a Cloud Spanner instance. Configurations
                   define the geographic placement of nodes and their
                   replication.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [instance_config, update_mask]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.UpdateInstanceConfigRequest):
            request = spanner_instance_admin.UpdateInstanceConfigRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if instance_config is not None:
                request.instance_config = instance_config
            if update_mask is not None:
                request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_instance_config]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("instance_config.name", request.instance_config.name),)
            ),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.InstanceConfig,
            metadata_type=spanner_instance_admin.UpdateInstanceConfigMetadata,
        )

        # Done; return the response.
        return response

    def delete_instance_config(
        self,
        request: Optional[
            Union[spanner_instance_admin.DeleteInstanceConfigRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes the instance configuration. Deletion is only allowed
        when no instances are using the configuration. If any instances
        are using the configuration, returns ``FAILED_PRECONDITION``.

        Only user-managed configurations can be deleted.

        Authorization requires ``spanner.instanceConfigs.delete``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstanceConfig.name].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_delete_instance_config():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.DeleteInstanceConfigRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_instance_config(request=request)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.DeleteInstanceConfigRequest, dict]):
                The request object. The request for
                [DeleteInstanceConfig][google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstanceConfig].
            name (str):
                Required. The name of the instance configuration to be
                deleted. Values are of the form
                ``projects/<project>/instanceConfigs/<instance_config>``

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.DeleteInstanceConfigRequest):
            request = spanner_instance_admin.DeleteInstanceConfigRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_instance_config]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def list_instance_config_operations(
        self,
        request: Optional[
            Union[spanner_instance_admin.ListInstanceConfigOperationsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListInstanceConfigOperationsPager:
        r"""Lists the user-managed instance configuration long-running
        operations in the given project. An instance configuration
        operation has a name of the form
        ``projects/<project>/instanceConfigs/<instance_config>/operations/<operation>``.
        The long-running operation metadata field type
        ``metadata.type_url`` describes the type of the metadata.
        Operations returned include those that have
        completed/failed/canceled within the last 7 days, and pending
        operations. Operations returned are ordered by
        ``operation.metadata.value.start_time`` in descending order
        starting from the most recently started operation.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_list_instance_config_operations():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.ListInstanceConfigOperationsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_instance_config_operations(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.ListInstanceConfigOperationsRequest, dict]):
                The request object. The request for
                [ListInstanceConfigOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigOperations].
            parent (str):
                Required. The project of the instance configuration
                operations. Values are of the form
                ``projects/<project>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.services.instance_admin.pagers.ListInstanceConfigOperationsPager:
                The response for
                   [ListInstanceConfigOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigOperations].

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, spanner_instance_admin.ListInstanceConfigOperationsRequest
        ):
            request = spanner_instance_admin.ListInstanceConfigOperationsRequest(
                request
            )
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.list_instance_config_operations
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListInstanceConfigOperationsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def list_instances(
        self,
        request: Optional[
            Union[spanner_instance_admin.ListInstancesRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListInstancesPager:
        r"""Lists all instances in the given project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_list_instances():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.ListInstancesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_instances(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.ListInstancesRequest, dict]):
                The request object. The request for
                [ListInstances][google.spanner.admin.instance.v1.InstanceAdmin.ListInstances].
            parent (str):
                Required. The name of the project for which a list of
                instances is requested. Values are of the form
                ``projects/<project>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.services.instance_admin.pagers.ListInstancesPager:
                The response for
                   [ListInstances][google.spanner.admin.instance.v1.InstanceAdmin.ListInstances].

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.ListInstancesRequest):
            request = spanner_instance_admin.ListInstancesRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_instances]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListInstancesPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def list_instance_partitions(
        self,
        request: Optional[
            Union[spanner_instance_admin.ListInstancePartitionsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListInstancePartitionsPager:
        r"""Lists all instance partitions for the given instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_list_instance_partitions():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.ListInstancePartitionsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_instance_partitions(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionsRequest, dict]):
                The request object. The request for
                [ListInstancePartitions][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitions].
            parent (str):
                Required. The instance whose instance partitions should
                be listed. Values are of the form
                ``projects/<project>/instances/<instance>``. Use
                ``{instance} = '-'`` to list instance partitions for all
                Instances in a project, e.g.,
                ``projects/myproject/instances/-``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.services.instance_admin.pagers.ListInstancePartitionsPager:
                The response for
                   [ListInstancePartitions][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitions].

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, spanner_instance_admin.ListInstancePartitionsRequest
        ):
            request = spanner_instance_admin.ListInstancePartitionsRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_instance_partitions]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListInstancePartitionsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_instance(
        self,
        request: Optional[
            Union[spanner_instance_admin.GetInstanceRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> spanner_instance_admin.Instance:
        r"""Gets information about a particular instance.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_get_instance():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.GetInstanceRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_instance(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.GetInstanceRequest, dict]):
                The request object. The request for
                [GetInstance][google.spanner.admin.instance.v1.InstanceAdmin.GetInstance].
            name (str):
                Required. The name of the requested instance. Values are
                of the form ``projects/<project>/instances/<instance>``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.types.Instance:
                An isolated set of Cloud Spanner
                resources on which databases can be
                hosted.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.GetInstanceRequest):
            request = spanner_instance_admin.GetInstanceRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_instance]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def create_instance(
        self,
        request: Optional[
            Union[spanner_instance_admin.CreateInstanceRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        instance_id: Optional[str] = None,
        instance: Optional[spanner_instance_admin.Instance] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Creates an instance and begins preparing it to begin serving.
        The returned long-running operation can be used to track the
        progress of preparing the new instance. The instance name is
        assigned by the caller. If the named instance already exists,
        ``CreateInstance`` returns ``ALREADY_EXISTS``.

        Immediately upon completion of this request:

        - The instance is readable via the API, with all requested
          attributes but no allocated resources. Its state is
          ``CREATING``.

        Until completion of the returned operation:

        - Cancelling the operation renders the instance immediately
          unreadable via the API.
        - The instance can be deleted.
        - All other attempts to modify the instance are rejected.

        Upon completion of the returned operation:

        - Billing for all successfully-allocated resources begins (some
          types may have lower than the requested levels).
        - Databases can be created in the instance.
        - The instance's allocated resource levels are readable via the
          API.
        - The instance's state becomes ``READY``.

        The returned long-running operation will have a name of the
        format ``<instance_name>/operations/<operation_id>`` and can be
        used to track creation of the instance. The metadata field type
        is
        [CreateInstanceMetadata][google.spanner.admin.instance.v1.CreateInstanceMetadata].
        The response field type is
        [Instance][google.spanner.admin.instance.v1.Instance], if
        successful.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_create_instance():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                instance = spanner_admin_instance_v1.Instance()
                instance.name = "name_value"
                instance.config = "config_value"
                instance.display_name = "display_name_value"

                request = spanner_admin_instance_v1.CreateInstanceRequest(
                    parent="parent_value",
                    instance_id="instance_id_value",
                    instance=instance,
                )

                # Make the request
                operation = client.create_instance(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.CreateInstanceRequest, dict]):
                The request object. The request for
                [CreateInstance][google.spanner.admin.instance.v1.InstanceAdmin.CreateInstance].
            parent (str):
                Required. The name of the project in which to create the
                instance. Values are of the form ``projects/<project>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance_id (str):
                Required. The ID of the instance to create. Valid
                identifiers are of the form ``[a-z][-a-z0-9]*[a-z0-9]``
                and must be between 2 and 64 characters in length.

                This corresponds to the ``instance_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance (google.cloud.spanner_admin_instance_v1.types.Instance):
                Required. The instance to create. The name may be
                omitted, but if specified must be
                ``<parent>/instances/<instance_id>``.

                This corresponds to the ``instance`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.spanner_admin_instance_v1.types.Instance`
                An isolated set of Cloud Spanner resources on which
                databases can be hosted.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, instance_id, instance]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.CreateInstanceRequest):
            request = spanner_instance_admin.CreateInstanceRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent
            if instance_id is not None:
                request.instance_id = instance_id
            if instance is not None:
                request.instance = instance

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_instance]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.Instance,
            metadata_type=spanner_instance_admin.CreateInstanceMetadata,
        )

        # Done; return the response.
        return response

    def update_instance(
        self,
        request: Optional[
            Union[spanner_instance_admin.UpdateInstanceRequest, dict]
        ] = None,
        *,
        instance: Optional[spanner_instance_admin.Instance] = None,
        field_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Updates an instance, and begins allocating or releasing
        resources as requested. The returned long-running operation can
        be used to track the progress of updating the instance. If the
        named instance does not exist, returns ``NOT_FOUND``.

        Immediately upon completion of this request:

        - For resource types for which a decrease in the instance's
          allocation has been requested, billing is based on the
          newly-requested level.

        Until completion of the returned operation:

        - Cancelling the operation sets its metadata's
          [cancel_time][google.spanner.admin.instance.v1.UpdateInstanceMetadata.cancel_time],
          and begins restoring resources to their pre-request values.
          The operation is guaranteed to succeed at undoing all resource
          changes, after which point it terminates with a ``CANCELLED``
          status.
        - All other attempts to modify the instance are rejected.
        - Reading the instance via the API continues to give the
          pre-request resource levels.

        Upon completion of the returned operation:

        - Billing begins for all successfully-allocated resources (some
          types may have lower than the requested levels).
        - All newly-reserved resources are available for serving the
          instance's tables.
        - The instance's new resource levels are readable via the API.

        The returned long-running operation will have a name of the
        format ``<instance_name>/operations/<operation_id>`` and can be
        used to track the instance modification. The metadata field type
        is
        [UpdateInstanceMetadata][google.spanner.admin.instance.v1.UpdateInstanceMetadata].
        The response field type is
        [Instance][google.spanner.admin.instance.v1.Instance], if
        successful.

        Authorization requires ``spanner.instances.update`` permission
        on the resource
        [name][google.spanner.admin.instance.v1.Instance.name].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_update_instance():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                instance = spanner_admin_instance_v1.Instance()
                instance.name = "name_value"
                instance.config = "config_value"
                instance.display_name = "display_name_value"

                request = spanner_admin_instance_v1.UpdateInstanceRequest(
                    instance=instance,
                )

                # Make the request
                operation = client.update_instance(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.UpdateInstanceRequest, dict]):
                The request object. The request for
                [UpdateInstance][google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstance].
            instance (google.cloud.spanner_admin_instance_v1.types.Instance):
                Required. The instance to update, which must always
                include the instance name. Otherwise, only fields
                mentioned in
                [field_mask][google.spanner.admin.instance.v1.UpdateInstanceRequest.field_mask]
                need be included.

                This corresponds to the ``instance`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            field_mask (google.protobuf.field_mask_pb2.FieldMask):
                Required. A mask specifying which fields in
                [Instance][google.spanner.admin.instance.v1.Instance]
                should be updated. The field mask must always be
                specified; this prevents any future fields in
                [Instance][google.spanner.admin.instance.v1.Instance]
                from being erased accidentally by clients that do not
                know about them.

                This corresponds to the ``field_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.spanner_admin_instance_v1.types.Instance`
                An isolated set of Cloud Spanner resources on which
                databases can be hosted.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [instance, field_mask]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.UpdateInstanceRequest):
            request = spanner_instance_admin.UpdateInstanceRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if instance is not None:
                request.instance = instance
            if field_mask is not None:
                request.field_mask = field_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_instance]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("instance.name", request.instance.name),)
            ),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.Instance,
            metadata_type=spanner_instance_admin.UpdateInstanceMetadata,
        )

        # Done; return the response.
        return response

    def delete_instance(
        self,
        request: Optional[
            Union[spanner_instance_admin.DeleteInstanceRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes an instance.

        Immediately upon completion of the request:

        - Billing ceases for all of the instance's reserved resources.

        Soon afterward:

        - The instance and *all of its databases* immediately and
          irrevocably disappear from the API. All data in the databases
          is permanently deleted.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_delete_instance():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.DeleteInstanceRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_instance(request=request)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.DeleteInstanceRequest, dict]):
                The request object. The request for
                [DeleteInstance][google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstance].
            name (str):
                Required. The name of the instance to be deleted. Values
                are of the form
                ``projects/<project>/instances/<instance>``

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.DeleteInstanceRequest):
            request = spanner_instance_admin.DeleteInstanceRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_instance]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def set_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.SetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Sets the access control policy on an instance resource. Replaces
        any existing policy.

        Authorization requires ``spanner.instances.setIamPolicy`` on
        [resource][google.iam.v1.SetIamPolicyRequest.resource].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            def sample_set_iam_policy():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.SetIamPolicyRequest(
                    resource="resource_value",
                )

                # Make the request
                response = client.set_iam_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.iam.v1.iam_policy_pb2.SetIamPolicyRequest, dict]):
                The request object. Request message for ``SetIamPolicy`` method.
            resource (str):
                REQUIRED: The resource for which the
                policy is being specified. See the
                operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.iam.v1.policy_pb2.Policy:
                An Identity and Access Management (IAM) policy, which specifies access
                   controls for Google Cloud resources.

                   A Policy is a collection of bindings. A binding binds
                   one or more members, or principals, to a single role.
                   Principals can be user accounts, service accounts,
                   Google groups, and domains (such as G Suite). A role
                   is a named list of permissions; each role can be an
                   IAM predefined role or a user-created custom role.

                   For some types of Google Cloud resources, a binding
                   can also specify a condition, which is a logical
                   expression that allows access to a resource only if
                   the expression evaluates to true. A condition can add
                   constraints based on attributes of the request, the
                   resource, or both. To learn which resources support
                   conditions in their IAM policies, see the [IAM
                   documentation](https://cloud.google.com/iam/help/conditions/resource-policies).

                   **JSON example:**

                   :literal:``     {       "bindings": [         {           "role": "roles/resourcemanager.organizationAdmin",           "members": [             "user:mike@example.com",             "group:admins@example.com",             "domain:google.com",             "serviceAccount:my-project-id@appspot.gserviceaccount.com"           ]         },         {           "role": "roles/resourcemanager.organizationViewer",           "members": [             "user:eve@example.com"           ],           "condition": {             "title": "expirable access",             "description": "Does not grant access after Sep 2020",             "expression": "request.time <             timestamp('2020-10-01T00:00:00.000Z')",           }         }       ],       "etag": "BwWWja0YfJA=",       "version": 3     }`\ \`

                   **YAML example:**

                   :literal:``     bindings:     - members:       - user:mike@example.com       - group:admins@example.com       - domain:google.com       - serviceAccount:my-project-id@appspot.gserviceaccount.com       role: roles/resourcemanager.organizationAdmin     - members:       - user:eve@example.com       role: roles/resourcemanager.organizationViewer       condition:         title: expirable access         description: Does not grant access after Sep 2020         expression: request.time < timestamp('2020-10-01T00:00:00.000Z')     etag: BwWWja0YfJA=     version: 3`\ \`

                   For a description of IAM and its features, see the
                   [IAM
                   documentation](https://cloud.google.com/iam/docs/).

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        if isinstance(request, dict):
            # - The request isn't a proto-plus wrapped type,
            #   so it must be constructed via keyword expansion.
            request = iam_policy_pb2.SetIamPolicyRequest(**request)
        elif not request:
            # Null request, just make one.
            request = iam_policy_pb2.SetIamPolicyRequest()
            if resource is not None:
                request.resource = resource

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.set_iam_policy]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_iam_policy(
        self,
        request: Optional[Union[iam_policy_pb2.GetIamPolicyRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_pb2.Policy:
        r"""Gets the access control policy for an instance resource. Returns
        an empty policy if an instance exists but does not have a policy
        set.

        Authorization requires ``spanner.instances.getIamPolicy`` on
        [resource][google.iam.v1.GetIamPolicyRequest.resource].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            def sample_get_iam_policy():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.GetIamPolicyRequest(
                    resource="resource_value",
                )

                # Make the request
                response = client.get_iam_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.iam.v1.iam_policy_pb2.GetIamPolicyRequest, dict]):
                The request object. Request message for ``GetIamPolicy`` method.
            resource (str):
                REQUIRED: The resource for which the
                policy is being requested. See the
                operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.iam.v1.policy_pb2.Policy:
                An Identity and Access Management (IAM) policy, which specifies access
                   controls for Google Cloud resources.

                   A Policy is a collection of bindings. A binding binds
                   one or more members, or principals, to a single role.
                   Principals can be user accounts, service accounts,
                   Google groups, and domains (such as G Suite). A role
                   is a named list of permissions; each role can be an
                   IAM predefined role or a user-created custom role.

                   For some types of Google Cloud resources, a binding
                   can also specify a condition, which is a logical
                   expression that allows access to a resource only if
                   the expression evaluates to true. A condition can add
                   constraints based on attributes of the request, the
                   resource, or both. To learn which resources support
                   conditions in their IAM policies, see the [IAM
                   documentation](https://cloud.google.com/iam/help/conditions/resource-policies).

                   **JSON example:**

                   :literal:``     {       "bindings": [         {           "role": "roles/resourcemanager.organizationAdmin",           "members": [             "user:mike@example.com",             "group:admins@example.com",             "domain:google.com",             "serviceAccount:my-project-id@appspot.gserviceaccount.com"           ]         },         {           "role": "roles/resourcemanager.organizationViewer",           "members": [             "user:eve@example.com"           ],           "condition": {             "title": "expirable access",             "description": "Does not grant access after Sep 2020",             "expression": "request.time <             timestamp('2020-10-01T00:00:00.000Z')",           }         }       ],       "etag": "BwWWja0YfJA=",       "version": 3     }`\ \`

                   **YAML example:**

                   :literal:``     bindings:     - members:       - user:mike@example.com       - group:admins@example.com       - domain:google.com       - serviceAccount:my-project-id@appspot.gserviceaccount.com       role: roles/resourcemanager.organizationAdmin     - members:       - user:eve@example.com       role: roles/resourcemanager.organizationViewer       condition:         title: expirable access         description: Does not grant access after Sep 2020         expression: request.time < timestamp('2020-10-01T00:00:00.000Z')     etag: BwWWja0YfJA=     version: 3`\ \`

                   For a description of IAM and its features, see the
                   [IAM
                   documentation](https://cloud.google.com/iam/docs/).

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        if isinstance(request, dict):
            # - The request isn't a proto-plus wrapped type,
            #   so it must be constructed via keyword expansion.
            request = iam_policy_pb2.GetIamPolicyRequest(**request)
        elif not request:
            # Null request, just make one.
            request = iam_policy_pb2.GetIamPolicyRequest()
            if resource is not None:
                request.resource = resource

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_iam_policy]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def test_iam_permissions(
        self,
        request: Optional[Union[iam_policy_pb2.TestIamPermissionsRequest, dict]] = None,
        *,
        resource: Optional[str] = None,
        permissions: Optional[MutableSequence[str]] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        r"""Returns permissions that the caller has on the specified
        instance resource.

        Attempting this RPC on a non-existent Cloud Spanner instance
        resource will result in a NOT_FOUND error if the user has
        ``spanner.instances.list`` permission on the containing Google
        Cloud Project. Otherwise returns an empty set of permissions.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1
            from google.iam.v1 import iam_policy_pb2  # type: ignore

            def sample_test_iam_permissions():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = iam_policy_pb2.TestIamPermissionsRequest(
                    resource="resource_value",
                    permissions=['permissions_value1', 'permissions_value2'],
                )

                # Make the request
                response = client.test_iam_permissions(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.iam.v1.iam_policy_pb2.TestIamPermissionsRequest, dict]):
                The request object. Request message for ``TestIamPermissions`` method.
            resource (str):
                REQUIRED: The resource for which the
                policy detail is being requested. See
                the operation documentation for the
                appropriate value for this field.

                This corresponds to the ``resource`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            permissions (MutableSequence[str]):
                The set of permissions to check for the ``resource``.
                Permissions with wildcards (such as '*' or 'storage.*')
                are not allowed. For more information see `IAM
                Overview <https://cloud.google.com/iam/docs/overview#permissions>`__.

                This corresponds to the ``permissions`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.iam.v1.iam_policy_pb2.TestIamPermissionsResponse:
                Response message for TestIamPermissions method.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [resource, permissions]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        if isinstance(request, dict):
            # - The request isn't a proto-plus wrapped type,
            #   so it must be constructed via keyword expansion.
            request = iam_policy_pb2.TestIamPermissionsRequest(**request)
        elif not request:
            # Null request, just make one.
            request = iam_policy_pb2.TestIamPermissionsRequest()
            if resource is not None:
                request.resource = resource
            if permissions:
                request.permissions.extend(permissions)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.test_iam_permissions]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("resource", request.resource),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_instance_partition(
        self,
        request: Optional[
            Union[spanner_instance_admin.GetInstancePartitionRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> spanner_instance_admin.InstancePartition:
        r"""Gets information about a particular instance
        partition.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_get_instance_partition():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.GetInstancePartitionRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_instance_partition(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.GetInstancePartitionRequest, dict]):
                The request object. The request for
                [GetInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.GetInstancePartition].
            name (str):
                Required. The name of the requested instance partition.
                Values are of the form
                ``projects/{project}/instances/{instance}/instancePartitions/{instance_partition}``.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.types.InstancePartition:
                An isolated set of Cloud Spanner
                resources that databases can define
                placements on.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.GetInstancePartitionRequest):
            request = spanner_instance_admin.GetInstancePartitionRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_instance_partition]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def create_instance_partition(
        self,
        request: Optional[
            Union[spanner_instance_admin.CreateInstancePartitionRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        instance_partition: Optional[spanner_instance_admin.InstancePartition] = None,
        instance_partition_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Creates an instance partition and begins preparing it to be
        used. The returned long-running operation can be used to track
        the progress of preparing the new instance partition. The
        instance partition name is assigned by the caller. If the named
        instance partition already exists, ``CreateInstancePartition``
        returns ``ALREADY_EXISTS``.

        Immediately upon completion of this request:

        - The instance partition is readable via the API, with all
          requested attributes but no allocated resources. Its state is
          ``CREATING``.

        Until completion of the returned operation:

        - Cancelling the operation renders the instance partition
          immediately unreadable via the API.
        - The instance partition can be deleted.
        - All other attempts to modify the instance partition are
          rejected.

        Upon completion of the returned operation:

        - Billing for all successfully-allocated resources begins (some
          types may have lower than the requested levels).
        - Databases can start using this instance partition.
        - The instance partition's allocated resource levels are
          readable via the API.
        - The instance partition's state becomes ``READY``.

        The returned long-running operation will have a name of the
        format ``<instance_partition_name>/operations/<operation_id>``
        and can be used to track creation of the instance partition. The
        metadata field type is
        [CreateInstancePartitionMetadata][google.spanner.admin.instance.v1.CreateInstancePartitionMetadata].
        The response field type is
        [InstancePartition][google.spanner.admin.instance.v1.InstancePartition],
        if successful.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_create_instance_partition():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                instance_partition = spanner_admin_instance_v1.InstancePartition()
                instance_partition.node_count = 1070
                instance_partition.name = "name_value"
                instance_partition.config = "config_value"
                instance_partition.display_name = "display_name_value"

                request = spanner_admin_instance_v1.CreateInstancePartitionRequest(
                    parent="parent_value",
                    instance_partition_id="instance_partition_id_value",
                    instance_partition=instance_partition,
                )

                # Make the request
                operation = client.create_instance_partition(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.CreateInstancePartitionRequest, dict]):
                The request object. The request for
                [CreateInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.CreateInstancePartition].
            parent (str):
                Required. The name of the instance in which to create
                the instance partition. Values are of the form
                ``projects/<project>/instances/<instance>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance_partition (google.cloud.spanner_admin_instance_v1.types.InstancePartition):
                Required. The instance partition to create. The
                instance_partition.name may be omitted, but if specified
                must be
                ``<parent>/instancePartitions/<instance_partition_id>``.

                This corresponds to the ``instance_partition`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            instance_partition_id (str):
                Required. The ID of the instance partition to create.
                Valid identifiers are of the form
                ``[a-z][-a-z0-9]*[a-z0-9]`` and must be between 2 and 64
                characters in length.

                This corresponds to the ``instance_partition_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.spanner_admin_instance_v1.types.InstancePartition` An isolated set of Cloud Spanner resources that databases can define
                   placements on.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, instance_partition, instance_partition_id]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, spanner_instance_admin.CreateInstancePartitionRequest
        ):
            request = spanner_instance_admin.CreateInstancePartitionRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent
            if instance_partition is not None:
                request.instance_partition = instance_partition
            if instance_partition_id is not None:
                request.instance_partition_id = instance_partition_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.create_instance_partition
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.InstancePartition,
            metadata_type=spanner_instance_admin.CreateInstancePartitionMetadata,
        )

        # Done; return the response.
        return response

    def delete_instance_partition(
        self,
        request: Optional[
            Union[spanner_instance_admin.DeleteInstancePartitionRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes an existing instance partition. Requires that the
        instance partition is not used by any database or backup and is
        not the default instance partition of an instance.

        Authorization requires ``spanner.instancePartitions.delete``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstancePartition.name].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_delete_instance_partition():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.DeleteInstancePartitionRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_instance_partition(request=request)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.DeleteInstancePartitionRequest, dict]):
                The request object. The request for
                [DeleteInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstancePartition].
            name (str):
                Required. The name of the instance partition to be
                deleted. Values are of the form
                ``projects/{project}/instances/{instance}/instancePartitions/{instance_partition}``

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, spanner_instance_admin.DeleteInstancePartitionRequest
        ):
            request = spanner_instance_admin.DeleteInstancePartitionRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if name is not None:
                request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.delete_instance_partition
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def update_instance_partition(
        self,
        request: Optional[
            Union[spanner_instance_admin.UpdateInstancePartitionRequest, dict]
        ] = None,
        *,
        instance_partition: Optional[spanner_instance_admin.InstancePartition] = None,
        field_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Updates an instance partition, and begins allocating or
        releasing resources as requested. The returned long-running
        operation can be used to track the progress of updating the
        instance partition. If the named instance partition does not
        exist, returns ``NOT_FOUND``.

        Immediately upon completion of this request:

        - For resource types for which a decrease in the instance
          partition's allocation has been requested, billing is based on
          the newly-requested level.

        Until completion of the returned operation:

        - Cancelling the operation sets its metadata's
          [cancel_time][google.spanner.admin.instance.v1.UpdateInstancePartitionMetadata.cancel_time],
          and begins restoring resources to their pre-request values.
          The operation is guaranteed to succeed at undoing all resource
          changes, after which point it terminates with a ``CANCELLED``
          status.
        - All other attempts to modify the instance partition are
          rejected.
        - Reading the instance partition via the API continues to give
          the pre-request resource levels.

        Upon completion of the returned operation:

        - Billing begins for all successfully-allocated resources (some
          types may have lower than the requested levels).
        - All newly-reserved resources are available for serving the
          instance partition's tables.
        - The instance partition's new resource levels are readable via
          the API.

        The returned long-running operation will have a name of the
        format ``<instance_partition_name>/operations/<operation_id>``
        and can be used to track the instance partition modification.
        The metadata field type is
        [UpdateInstancePartitionMetadata][google.spanner.admin.instance.v1.UpdateInstancePartitionMetadata].
        The response field type is
        [InstancePartition][google.spanner.admin.instance.v1.InstancePartition],
        if successful.

        Authorization requires ``spanner.instancePartitions.update``
        permission on the resource
        [name][google.spanner.admin.instance.v1.InstancePartition.name].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_update_instance_partition():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                instance_partition = spanner_admin_instance_v1.InstancePartition()
                instance_partition.node_count = 1070
                instance_partition.name = "name_value"
                instance_partition.config = "config_value"
                instance_partition.display_name = "display_name_value"

                request = spanner_admin_instance_v1.UpdateInstancePartitionRequest(
                    instance_partition=instance_partition,
                )

                # Make the request
                operation = client.update_instance_partition(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.UpdateInstancePartitionRequest, dict]):
                The request object. The request for
                [UpdateInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstancePartition].
            instance_partition (google.cloud.spanner_admin_instance_v1.types.InstancePartition):
                Required. The instance partition to update, which must
                always include the instance partition name. Otherwise,
                only fields mentioned in
                [field_mask][google.spanner.admin.instance.v1.UpdateInstancePartitionRequest.field_mask]
                need be included.

                This corresponds to the ``instance_partition`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            field_mask (google.protobuf.field_mask_pb2.FieldMask):
                Required. A mask specifying which fields in
                [InstancePartition][google.spanner.admin.instance.v1.InstancePartition]
                should be updated. The field mask must always be
                specified; this prevents any future fields in
                [InstancePartition][google.spanner.admin.instance.v1.InstancePartition]
                from being erased accidentally by clients that do not
                know about them.

                This corresponds to the ``field_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.spanner_admin_instance_v1.types.InstancePartition` An isolated set of Cloud Spanner resources that databases can define
                   placements on.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [instance_partition, field_mask]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, spanner_instance_admin.UpdateInstancePartitionRequest
        ):
            request = spanner_instance_admin.UpdateInstancePartitionRequest(request)
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if instance_partition is not None:
                request.instance_partition = instance_partition
            if field_mask is not None:
                request.field_mask = field_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.update_instance_partition
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("instance_partition.name", request.instance_partition.name),)
            ),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.InstancePartition,
            metadata_type=spanner_instance_admin.UpdateInstancePartitionMetadata,
        )

        # Done; return the response.
        return response

    def list_instance_partition_operations(
        self,
        request: Optional[
            Union[spanner_instance_admin.ListInstancePartitionOperationsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListInstancePartitionOperationsPager:
        r"""Lists instance partition long-running operations in the given
        instance. An instance partition operation has a name of the form
        ``projects/<project>/instances/<instance>/instancePartitions/<instance_partition>/operations/<operation>``.
        The long-running operation metadata field type
        ``metadata.type_url`` describes the type of the metadata.
        Operations returned include those that have
        completed/failed/canceled within the last 7 days, and pending
        operations. Operations returned are ordered by
        ``operation.metadata.value.start_time`` in descending order
        starting from the most recently started operation.

        Authorization requires
        ``spanner.instancePartitionOperations.list`` permission on the
        resource
        [parent][google.spanner.admin.instance.v1.ListInstancePartitionOperationsRequest.parent].

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_list_instance_partition_operations():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.ListInstancePartitionOperationsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_instance_partition_operations(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.ListInstancePartitionOperationsRequest, dict]):
                The request object. The request for
                [ListInstancePartitionOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitionOperations].
            parent (str):
                Required. The parent instance of the instance partition
                operations. Values are of the form
                ``projects/<project>/instances/<instance>``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.spanner_admin_instance_v1.services.instance_admin.pagers.ListInstancePartitionOperationsPager:
                The response for
                   [ListInstancePartitionOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitionOperations].

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request, spanner_instance_admin.ListInstancePartitionOperationsRequest
        ):
            request = spanner_instance_admin.ListInstancePartitionOperationsRequest(
                request
            )
            # If we have keyword arguments corresponding to fields on the
            # request, apply these.
            if parent is not None:
                request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[
            self._transport.list_instance_partition_operations
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListInstancePartitionOperationsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def move_instance(
        self,
        request: Optional[
            Union[spanner_instance_admin.MoveInstanceRequest, dict]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation.Operation:
        r"""Moves an instance to the target instance configuration. You can
        use the returned long-running operation to track the progress of
        moving the instance.

        ``MoveInstance`` returns ``FAILED_PRECONDITION`` if the instance
        meets any of the following criteria:

        - Is undergoing a move to a different instance configuration
        - Has backups
        - Has an ongoing update
        - Contains any CMEK-enabled databases
        - Is a free trial instance

        While the operation is pending:

        - All other attempts to modify the instance, including changes
          to its compute capacity, are rejected.

        - The following database and backup admin operations are
          rejected:

          - ``DatabaseAdmin.CreateDatabase``
          - ``DatabaseAdmin.UpdateDatabaseDdl`` (disabled if
            default_leader is specified in the request.)
          - ``DatabaseAdmin.RestoreDatabase``
          - ``DatabaseAdmin.CreateBackup``
          - ``DatabaseAdmin.CopyBackup``

        - Both the source and target instance configurations are subject
          to hourly compute and storage charges.

        - The instance might experience higher read-write latencies and
          a higher transaction abort rate. However, moving an instance
          doesn't cause any downtime.

        The returned long-running operation has a name of the format
        ``<instance_name>/operations/<operation_id>`` and can be used to
        track the move instance operation. The metadata field type is
        [MoveInstanceMetadata][google.spanner.admin.instance.v1.MoveInstanceMetadata].
        The response field type is
        [Instance][google.spanner.admin.instance.v1.Instance], if
        successful. Cancelling the operation sets its metadata's
        [cancel_time][google.spanner.admin.instance.v1.MoveInstanceMetadata.cancel_time].
        Cancellation is not immediate because it involves moving any
        data previously moved to the target instance configuration back
        to the original instance configuration. You can use this
        operation to track the progress of the cancellation. Upon
        successful completion of the cancellation, the operation
        terminates with ``CANCELLED`` status.

        If not cancelled, upon completion of the returned operation:

        - The instance successfully moves to the target instance
          configuration.
        - You are billed for compute and storage in target instance
          configuration.

        Authorization requires the ``spanner.instances.update``
        permission on the resource
        [instance][google.spanner.admin.instance.v1.Instance].

        For more details, see `Move an
        instance <https://cloud.google.com/spanner/docs/move-instance>`__.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import spanner_admin_instance_v1

            def sample_move_instance():
                # Create a client
                client = spanner_admin_instance_v1.InstanceAdminClient()

                # Initialize request argument(s)
                request = spanner_admin_instance_v1.MoveInstanceRequest(
                    name="name_value",
                    target_config="target_config_value",
                )

                # Make the request
                operation = client.move_instance(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.spanner_admin_instance_v1.types.MoveInstanceRequest, dict]):
                The request object. The request for
                [MoveInstance][google.spanner.admin.instance.v1.InstanceAdmin.MoveInstance].
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.spanner_admin_instance_v1.types.MoveInstanceResponse` The response for
                   [MoveInstance][google.spanner.admin.instance.v1.InstanceAdmin.MoveInstance].

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, spanner_instance_admin.MoveInstanceRequest):
            request = spanner_instance_admin.MoveInstanceRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.move_instance]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation.from_gapic(
            response,
            self._transport.operations_client,
            spanner_instance_admin.MoveInstanceResponse,
            metadata_type=spanner_instance_admin.MoveInstanceMetadata,
        )

        # Done; return the response.
        return response

    def __enter__(self) -> "InstanceAdminClient":
        return self

    def __exit__(self, type, value, traceback):
        """Releases underlying transport's resources.

        .. warning::
            ONLY use as a context manager if the transport is NOT shared
            with other clients! Exiting the with block will CLOSE the transport
            and may cause errors in other clients!
        """
        self.transport.close()

    def list_operations(
        self,
        request: Optional[operations_pb2.ListOperationsRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operations_pb2.ListOperationsResponse:
        r"""Lists operations that match the specified filter in the request.

        Args:
            request (:class:`~.operations_pb2.ListOperationsRequest`):
                The request object. Request message for
                `ListOperations` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            ~.operations_pb2.ListOperationsResponse:
                Response message for ``ListOperations`` method.
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.ListOperationsRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_operations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        try:
            # Send the request.
            response = rpc(
                request,
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            )

            # Done; return the response.
            return response
        except core_exceptions.GoogleAPICallError as e:
            self._add_cred_info_for_auth_errors(e)
            raise e

    def get_operation(
        self,
        request: Optional[operations_pb2.GetOperationRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operations_pb2.Operation:
        r"""Gets the latest state of a long-running operation.

        Args:
            request (:class:`~.operations_pb2.GetOperationRequest`):
                The request object. Request message for
                `GetOperation` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            ~.operations_pb2.Operation:
                An ``Operation`` object.
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.GetOperationRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        try:
            # Send the request.
            response = rpc(
                request,
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            )

            # Done; return the response.
            return response
        except core_exceptions.GoogleAPICallError as e:
            self._add_cred_info_for_auth_errors(e)
            raise e

    def delete_operation(
        self,
        request: Optional[operations_pb2.DeleteOperationRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes a long-running operation.

        This method indicates that the client is no longer interested
        in the operation result. It does not cancel the operation.
        If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.

        Args:
            request (:class:`~.operations_pb2.DeleteOperationRequest`):
                The request object. Request message for
                `DeleteOperation` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            None
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.DeleteOperationRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    def cancel_operation(
        self,
        request: Optional[operations_pb2.CancelOperationRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Starts asynchronous cancellation on a long-running operation.

        The server makes a best effort to cancel the operation, but success
        is not guaranteed.  If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.

        Args:
            request (:class:`~.operations_pb2.CancelOperationRequest`):
                The request object. Request message for
                `CancelOperation` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            None
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.CancelOperationRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.cancel_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__

__all__ = ("InstanceAdminClient",)
