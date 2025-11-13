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
import logging as std_logging
import re
from typing import (
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry_async as retries
from google.api_core.client_options import ClientOptions
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

from google.cloud.monitoring_v3 import gapic_version as package_version

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.api import monitored_resource_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore

from google.cloud.monitoring_v3.services.group_service import pagers
from google.cloud.monitoring_v3.types import group
from google.cloud.monitoring_v3.types import group as gm_group
from google.cloud.monitoring_v3.types import group_service

from .client import GroupServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, GroupServiceTransport
from .transports.grpc_asyncio import GroupServiceGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class GroupServiceAsyncClient:
    """The Group API lets you inspect and manage your
    `groups <#google.monitoring.v3.Group>`__.

    A group is a named filter that is used to identify a collection of
    monitored resources. Groups are typically used to mirror the
    physical and/or logical topology of the environment. Because group
    membership is computed dynamically, monitored resources that are
    started in the future are automatically placed in matching groups.
    By using a group to name monitored resources in, for example, an
    alert policy, the target of that alert policy is updated
    automatically as monitored resources are added and removed from the
    infrastructure.
    """

    _client: GroupServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = GroupServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = GroupServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = GroupServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = GroupServiceClient._DEFAULT_UNIVERSE

    group_path = staticmethod(GroupServiceClient.group_path)
    parse_group_path = staticmethod(GroupServiceClient.parse_group_path)
    common_billing_account_path = staticmethod(
        GroupServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        GroupServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(GroupServiceClient.common_folder_path)
    parse_common_folder_path = staticmethod(GroupServiceClient.parse_common_folder_path)
    common_organization_path = staticmethod(GroupServiceClient.common_organization_path)
    parse_common_organization_path = staticmethod(
        GroupServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(GroupServiceClient.common_project_path)
    parse_common_project_path = staticmethod(
        GroupServiceClient.parse_common_project_path
    )
    common_location_path = staticmethod(GroupServiceClient.common_location_path)
    parse_common_location_path = staticmethod(
        GroupServiceClient.parse_common_location_path
    )

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            GroupServiceAsyncClient: The constructed client.
        """
        return GroupServiceClient.from_service_account_info.__func__(GroupServiceAsyncClient, info, *args, **kwargs)  # type: ignore

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
            GroupServiceAsyncClient: The constructed client.
        """
        return GroupServiceClient.from_service_account_file.__func__(GroupServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

    from_service_account_json = from_service_account_file

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[ClientOptions] = None
    ):
        """Return the API endpoint and client cert source for mutual TLS.

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
        return GroupServiceClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> GroupServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            GroupServiceTransport: The transport used by the client instance.
        """
        return self._client.transport

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._client._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used
                by the client instance.
        """
        return self._client._universe_domain

    get_transport_class = GroupServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, GroupServiceTransport, Callable[..., GroupServiceTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the group service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,GroupServiceTransport,Callable[..., GroupServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the GroupServiceTransport constructor.
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
                default "googleapis.com" universe. Note that ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        self._client = GroupServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.monitoring_v3.GroupServiceAsyncClient`.",
                extra={
                    "serviceName": "google.monitoring.v3.GroupService",
                    "universeDomain": getattr(
                        self._client._transport._credentials, "universe_domain", ""
                    ),
                    "credentialsType": f"{type(self._client._transport._credentials).__module__}.{type(self._client._transport._credentials).__qualname__}",
                    "credentialsInfo": getattr(
                        self.transport._credentials, "get_cred_info", lambda: None
                    )(),
                }
                if hasattr(self._client._transport, "_credentials")
                else {
                    "serviceName": "google.monitoring.v3.GroupService",
                    "credentialsType": None,
                },
            )

    async def list_groups(
        self,
        request: Optional[Union[group_service.ListGroupsRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListGroupsAsyncPager:
        r"""Lists the existing groups.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_groups():
                # Create a client
                client = monitoring_v3.GroupServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListGroupsRequest(
                    children_of_group="children_of_group_value",
                    name="name_value",
                )

                # Make the request
                page_result = client.list_groups(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListGroupsRequest, dict]]):
                The request object. The ``ListGroup`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                whose groups are to be listed. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.monitoring_v3.services.group_service.pagers.ListGroupsAsyncPager:
                The ListGroups response.

                Iterating over this object will yield results and
                resolve additional pages automatically.

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
        if not isinstance(request, group_service.ListGroupsRequest):
            request = group_service.ListGroupsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_groups
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListGroupsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_group(
        self,
        request: Optional[Union[group_service.GetGroupRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> group.Group:
        r"""Gets a single group.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_get_group():
                # Create a client
                client = monitoring_v3.GroupServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.GetGroupRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_group(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.GetGroupRequest, dict]]):
                The request object. The ``GetGroup`` request.
            name (:class:`str`):
                Required. The group to retrieve. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.monitoring_v3.types.Group:
                The description of a dynamic collection of monitored resources. Each group
                   has a filter that is matched against monitored
                   resources and their associated metadata. If a group's
                   filter matches an available monitored resource, then
                   that resource is a member of that group. Groups can
                   contain any number of monitored resources, and each
                   monitored resource can be a member of any number of
                   groups.

                   Groups can be nested in parent-child hierarchies. The
                   parentName field identifies an optional parent for
                   each group. If a group has a parent, then the only
                   monitored resources available to be matched by the
                   group's filter are the resources contained in the
                   parent group. In other words, a group contains the
                   monitored resources that match its filter and the
                   filters of all the group's ancestors. A group without
                   a parent can contain any monitored resource.

                   For example, consider an infrastructure running a set
                   of instances with two user-defined tags:
                   "environment" and "role". A parent group has a
                   filter, environment="production". A child of that
                   parent group has a filter, role="transcoder". The
                   parent group contains all instances in the production
                   environment, regardless of their roles. The child
                   group contains instances that have the transcoder
                   role *and* are in the production environment.

                   The monitored resources contained in a group can
                   change at any moment, depending on what resources
                   exist and what filters are associated with the group
                   and its ancestors.

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
        if not isinstance(request, group_service.GetGroupRequest):
            request = group_service.GetGroupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_group
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def create_group(
        self,
        request: Optional[Union[group_service.CreateGroupRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        group: Optional[gm_group.Group] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> gm_group.Group:
        r"""Creates a new group.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_create_group():
                # Create a client
                client = monitoring_v3.GroupServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.CreateGroupRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.create_group(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.CreateGroupRequest, dict]]):
                The request object. The ``CreateGroup`` request.
            name (:class:`str`):
                Required. The
                `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
                in which to create the group. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            group (:class:`google.cloud.monitoring_v3.types.Group`):
                Required. A group definition. It is an error to define
                the ``name`` field because the system assigns the name.

                This corresponds to the ``group`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.monitoring_v3.types.Group:
                The description of a dynamic collection of monitored resources. Each group
                   has a filter that is matched against monitored
                   resources and their associated metadata. If a group's
                   filter matches an available monitored resource, then
                   that resource is a member of that group. Groups can
                   contain any number of monitored resources, and each
                   monitored resource can be a member of any number of
                   groups.

                   Groups can be nested in parent-child hierarchies. The
                   parentName field identifies an optional parent for
                   each group. If a group has a parent, then the only
                   monitored resources available to be matched by the
                   group's filter are the resources contained in the
                   parent group. In other words, a group contains the
                   monitored resources that match its filter and the
                   filters of all the group's ancestors. A group without
                   a parent can contain any monitored resource.

                   For example, consider an infrastructure running a set
                   of instances with two user-defined tags:
                   "environment" and "role". A parent group has a
                   filter, environment="production". A child of that
                   parent group has a filter, role="transcoder". The
                   parent group contains all instances in the production
                   environment, regardless of their roles. The child
                   group contains instances that have the transcoder
                   role *and* are in the production environment.

                   The monitored resources contained in a group can
                   change at any moment, depending on what resources
                   exist and what filters are associated with the group
                   and its ancestors.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name, group]
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
        if not isinstance(request, group_service.CreateGroupRequest):
            request = group_service.CreateGroupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name
        if group is not None:
            request.group = group

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_group
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_group(
        self,
        request: Optional[Union[group_service.UpdateGroupRequest, dict]] = None,
        *,
        group: Optional[gm_group.Group] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> gm_group.Group:
        r"""Updates an existing group. You can change any group attributes
        except ``name``.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_update_group():
                # Create a client
                client = monitoring_v3.GroupServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.UpdateGroupRequest(
                )

                # Make the request
                response = await client.update_group(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.UpdateGroupRequest, dict]]):
                The request object. The ``UpdateGroup`` request.
            group (:class:`google.cloud.monitoring_v3.types.Group`):
                Required. The new definition of the group. All fields of
                the existing group, excepting ``name``, are replaced
                with the corresponding fields of this group.

                This corresponds to the ``group`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.monitoring_v3.types.Group:
                The description of a dynamic collection of monitored resources. Each group
                   has a filter that is matched against monitored
                   resources and their associated metadata. If a group's
                   filter matches an available monitored resource, then
                   that resource is a member of that group. Groups can
                   contain any number of monitored resources, and each
                   monitored resource can be a member of any number of
                   groups.

                   Groups can be nested in parent-child hierarchies. The
                   parentName field identifies an optional parent for
                   each group. If a group has a parent, then the only
                   monitored resources available to be matched by the
                   group's filter are the resources contained in the
                   parent group. In other words, a group contains the
                   monitored resources that match its filter and the
                   filters of all the group's ancestors. A group without
                   a parent can contain any monitored resource.

                   For example, consider an infrastructure running a set
                   of instances with two user-defined tags:
                   "environment" and "role". A parent group has a
                   filter, environment="production". A child of that
                   parent group has a filter, role="transcoder". The
                   parent group contains all instances in the production
                   environment, regardless of their roles. The child
                   group contains instances that have the transcoder
                   role *and* are in the production environment.

                   The monitored resources contained in a group can
                   change at any moment, depending on what resources
                   exist and what filters are associated with the group
                   and its ancestors.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [group]
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
        if not isinstance(request, group_service.UpdateGroupRequest):
            request = group_service.UpdateGroupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if group is not None:
            request.group = group

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_group
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("group.name", request.group.name),)
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def delete_group(
        self,
        request: Optional[Union[group_service.DeleteGroupRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Deletes an existing group.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_delete_group():
                # Create a client
                client = monitoring_v3.GroupServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.DeleteGroupRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_group(request=request)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.DeleteGroupRequest, dict]]):
                The request object. The ``DeleteGroup`` request. The default behavior is to
                be able to delete a single group without any
                descendants.
            name (:class:`str`):
                Required. The group to delete. The format is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
        if not isinstance(request, group_service.DeleteGroupRequest):
            request = group_service.DeleteGroupRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_group
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    async def list_group_members(
        self,
        request: Optional[Union[group_service.ListGroupMembersRequest, dict]] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListGroupMembersAsyncPager:
        r"""Lists the monitored resources that are members of a
        group.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import monitoring_v3

            async def sample_list_group_members():
                # Create a client
                client = monitoring_v3.GroupServiceAsyncClient()

                # Initialize request argument(s)
                request = monitoring_v3.ListGroupMembersRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_group_members(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.monitoring_v3.types.ListGroupMembersRequest, dict]]):
                The request object. The ``ListGroupMembers`` request.
            name (:class:`str`):
                Required. The group whose members are listed. The format
                is:

                ::

                    projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.monitoring_v3.services.group_service.pagers.ListGroupMembersAsyncPager:
                The ListGroupMembers response.

                Iterating over this object will yield results and
                resolve additional pages automatically.

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
        if not isinstance(request, group_service.ListGroupMembersRequest):
            request = group_service.ListGroupMembersRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_group_members
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListGroupMembersAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def __aenter__(self) -> "GroupServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("GroupServiceAsyncClient",)
