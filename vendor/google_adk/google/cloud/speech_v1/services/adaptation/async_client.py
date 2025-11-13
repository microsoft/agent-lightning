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

from google.cloud.speech_v1 import gapic_version as package_version

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore

from google.cloud.speech_v1.services.adaptation import pagers
from google.cloud.speech_v1.types import cloud_speech_adaptation, resource

from .client import AdaptationClient
from .transports.base import DEFAULT_CLIENT_INFO, AdaptationTransport
from .transports.grpc_asyncio import AdaptationGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class AdaptationAsyncClient:
    """Service that implements Google Cloud Speech Adaptation API."""

    _client: AdaptationClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = AdaptationClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = AdaptationClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = AdaptationClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = AdaptationClient._DEFAULT_UNIVERSE

    custom_class_path = staticmethod(AdaptationClient.custom_class_path)
    parse_custom_class_path = staticmethod(AdaptationClient.parse_custom_class_path)
    phrase_set_path = staticmethod(AdaptationClient.phrase_set_path)
    parse_phrase_set_path = staticmethod(AdaptationClient.parse_phrase_set_path)
    common_billing_account_path = staticmethod(
        AdaptationClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        AdaptationClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(AdaptationClient.common_folder_path)
    parse_common_folder_path = staticmethod(AdaptationClient.parse_common_folder_path)
    common_organization_path = staticmethod(AdaptationClient.common_organization_path)
    parse_common_organization_path = staticmethod(
        AdaptationClient.parse_common_organization_path
    )
    common_project_path = staticmethod(AdaptationClient.common_project_path)
    parse_common_project_path = staticmethod(AdaptationClient.parse_common_project_path)
    common_location_path = staticmethod(AdaptationClient.common_location_path)
    parse_common_location_path = staticmethod(
        AdaptationClient.parse_common_location_path
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
            AdaptationAsyncClient: The constructed client.
        """
        return AdaptationClient.from_service_account_info.__func__(AdaptationAsyncClient, info, *args, **kwargs)  # type: ignore

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
            AdaptationAsyncClient: The constructed client.
        """
        return AdaptationClient.from_service_account_file.__func__(AdaptationAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return AdaptationClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> AdaptationTransport:
        """Returns the transport used by the client instance.

        Returns:
            AdaptationTransport: The transport used by the client instance.
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

    get_transport_class = AdaptationClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, AdaptationTransport, Callable[..., AdaptationTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the adaptation async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,AdaptationTransport,Callable[..., AdaptationTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the AdaptationTransport constructor.
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
        self._client = AdaptationClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.cloud.speech_v1.AdaptationAsyncClient`.",
                extra={
                    "serviceName": "google.cloud.speech.v1.Adaptation",
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
                    "serviceName": "google.cloud.speech.v1.Adaptation",
                    "credentialsType": None,
                },
            )

    async def create_phrase_set(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.CreatePhraseSetRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        phrase_set: Optional[resource.PhraseSet] = None,
        phrase_set_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> resource.PhraseSet:
        r"""Create a set of phrase hints. Each item in the set
        can be a single word or a multi-word phrase. The items
        in the PhraseSet are favored by the recognition model
        when you send a call that includes the PhraseSet.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_create_phrase_set():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.CreatePhraseSetRequest(
                    parent="parent_value",
                    phrase_set_id="phrase_set_id_value",
                )

                # Make the request
                response = await client.create_phrase_set(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.CreatePhraseSetRequest, dict]]):
                The request object. Message sent by the client for the ``CreatePhraseSet``
                method.
            parent (:class:`str`):
                Required. The parent resource where this phrase set will
                be created. Format:

                ``projects/{project}/locations/{location}``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            phrase_set (:class:`google.cloud.speech_v1.types.PhraseSet`):
                Required. The phrase set to create.
                This corresponds to the ``phrase_set`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            phrase_set_id (:class:`str`):
                Required. The ID to use for the
                phrase set, which will become the final
                component of the phrase set's resource
                name.

                This value should restrict to letters,
                numbers, and hyphens, with the first
                character a letter, the last a letter or
                a number, and be 4-63 characters.

                This corresponds to the ``phrase_set_id`` field
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
            google.cloud.speech_v1.types.PhraseSet:
                Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, phrase_set, phrase_set_id]
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
        if not isinstance(request, cloud_speech_adaptation.CreatePhraseSetRequest):
            request = cloud_speech_adaptation.CreatePhraseSetRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if phrase_set is not None:
            request.phrase_set = phrase_set
        if phrase_set_id is not None:
            request.phrase_set_id = phrase_set_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_phrase_set
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
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

    async def get_phrase_set(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.GetPhraseSetRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> resource.PhraseSet:
        r"""Get a phrase set.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_get_phrase_set():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.GetPhraseSetRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_phrase_set(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.GetPhraseSetRequest, dict]]):
                The request object. Message sent by the client for the ``GetPhraseSet``
                method.
            name (:class:`str`):
                Required. The name of the phrase set to retrieve.
                Format:

                ``projects/{project}/locations/{location}/phraseSets/{phrase_set}``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

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
            google.cloud.speech_v1.types.PhraseSet:
                Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

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
        if not isinstance(request, cloud_speech_adaptation.GetPhraseSetRequest):
            request = cloud_speech_adaptation.GetPhraseSetRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_phrase_set
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

    async def list_phrase_set(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.ListPhraseSetRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListPhraseSetAsyncPager:
        r"""List phrase sets.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_list_phrase_set():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.ListPhraseSetRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_phrase_set(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.ListPhraseSetRequest, dict]]):
                The request object. Message sent by the client for the ``ListPhraseSet``
                method.
            parent (:class:`str`):
                Required. The parent, which owns this collection of
                phrase set. Format:

                ``projects/{project}/locations/{location}``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

                This corresponds to the ``parent`` field
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
            google.cloud.speech_v1.services.adaptation.pagers.ListPhraseSetAsyncPager:
                Message returned to the client by the ListPhraseSet
                method.

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
        if not isinstance(request, cloud_speech_adaptation.ListPhraseSetRequest):
            request = cloud_speech_adaptation.ListPhraseSetRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_phrase_set
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
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
        response = pagers.ListPhraseSetAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_phrase_set(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.UpdatePhraseSetRequest, dict]
        ] = None,
        *,
        phrase_set: Optional[resource.PhraseSet] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> resource.PhraseSet:
        r"""Update a phrase set.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_update_phrase_set():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.UpdatePhraseSetRequest(
                )

                # Make the request
                response = await client.update_phrase_set(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.UpdatePhraseSetRequest, dict]]):
                The request object. Message sent by the client for the ``UpdatePhraseSet``
                method.
            phrase_set (:class:`google.cloud.speech_v1.types.PhraseSet`):
                Required. The phrase set to update.

                The phrase set's ``name`` field is used to identify the
                set to be updated. Format:

                ``projects/{project}/locations/{location}/phraseSets/{phrase_set}``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

                This corresponds to the ``phrase_set`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                The list of fields to be updated.
                This corresponds to the ``update_mask`` field
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
            google.cloud.speech_v1.types.PhraseSet:
                Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [phrase_set, update_mask]
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
        if not isinstance(request, cloud_speech_adaptation.UpdatePhraseSetRequest):
            request = cloud_speech_adaptation.UpdatePhraseSetRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if phrase_set is not None:
            request.phrase_set = phrase_set
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_phrase_set
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("phrase_set.name", request.phrase_set.name),)
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

    async def delete_phrase_set(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.DeletePhraseSetRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Delete a phrase set.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_delete_phrase_set():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.DeletePhraseSetRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_phrase_set(request=request)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.DeletePhraseSetRequest, dict]]):
                The request object. Message sent by the client for the ``DeletePhraseSet``
                method.
            name (:class:`str`):
                Required. The name of the phrase set to delete. Format:

                ``projects/{project}/locations/{location}/phraseSets/{phrase_set}``

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
        if not isinstance(request, cloud_speech_adaptation.DeletePhraseSetRequest):
            request = cloud_speech_adaptation.DeletePhraseSetRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_phrase_set
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

    async def create_custom_class(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.CreateCustomClassRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        custom_class: Optional[resource.CustomClass] = None,
        custom_class_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> resource.CustomClass:
        r"""Create a custom class.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_create_custom_class():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.CreateCustomClassRequest(
                    parent="parent_value",
                    custom_class_id="custom_class_id_value",
                )

                # Make the request
                response = await client.create_custom_class(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.CreateCustomClassRequest, dict]]):
                The request object. Message sent by the client for the ``CreateCustomClass``
                method.
            parent (:class:`str`):
                Required. The parent resource where this custom class
                will be created. Format:

                ``projects/{project}/locations/{location}/customClasses``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            custom_class (:class:`google.cloud.speech_v1.types.CustomClass`):
                Required. The custom class to create.
                This corresponds to the ``custom_class`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            custom_class_id (:class:`str`):
                Required. The ID to use for the
                custom class, which will become the
                final component of the custom class'
                resource name.

                This value should restrict to letters,
                numbers, and hyphens, with the first
                character a letter, the last a letter or
                a number, and be 4-63 characters.

                This corresponds to the ``custom_class_id`` field
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
            google.cloud.speech_v1.types.CustomClass:
                A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, custom_class, custom_class_id]
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
        if not isinstance(request, cloud_speech_adaptation.CreateCustomClassRequest):
            request = cloud_speech_adaptation.CreateCustomClassRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if custom_class is not None:
            request.custom_class = custom_class
        if custom_class_id is not None:
            request.custom_class_id = custom_class_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_custom_class
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
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

    async def get_custom_class(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.GetCustomClassRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> resource.CustomClass:
        r"""Get a custom class.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_get_custom_class():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.GetCustomClassRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_custom_class(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.GetCustomClassRequest, dict]]):
                The request object. Message sent by the client for the ``GetCustomClass``
                method.
            name (:class:`str`):
                Required. The name of the custom class to retrieve.
                Format:

                ``projects/{project}/locations/{location}/customClasses/{custom_class}``

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
            google.cloud.speech_v1.types.CustomClass:
                A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

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
        if not isinstance(request, cloud_speech_adaptation.GetCustomClassRequest):
            request = cloud_speech_adaptation.GetCustomClassRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_custom_class
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

    async def list_custom_classes(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.ListCustomClassesRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListCustomClassesAsyncPager:
        r"""List custom classes.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_list_custom_classes():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.ListCustomClassesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_custom_classes(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.ListCustomClassesRequest, dict]]):
                The request object. Message sent by the client for the ``ListCustomClasses``
                method.
            parent (:class:`str`):
                Required. The parent, which owns this collection of
                custom classes. Format:

                ``projects/{project}/locations/{location}/customClasses``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

                This corresponds to the ``parent`` field
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
            google.cloud.speech_v1.services.adaptation.pagers.ListCustomClassesAsyncPager:
                Message returned to the client by the ListCustomClasses
                method.

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
        if not isinstance(request, cloud_speech_adaptation.ListCustomClassesRequest):
            request = cloud_speech_adaptation.ListCustomClassesRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_custom_classes
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
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
        response = pagers.ListCustomClassesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_custom_class(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.UpdateCustomClassRequest, dict]
        ] = None,
        *,
        custom_class: Optional[resource.CustomClass] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> resource.CustomClass:
        r"""Update a custom class.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_update_custom_class():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.UpdateCustomClassRequest(
                )

                # Make the request
                response = await client.update_custom_class(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.UpdateCustomClassRequest, dict]]):
                The request object. Message sent by the client for the ``UpdateCustomClass``
                method.
            custom_class (:class:`google.cloud.speech_v1.types.CustomClass`):
                Required. The custom class to update.

                The custom class's ``name`` field is used to identify
                the custom class to be updated. Format:

                ``projects/{project}/locations/{location}/customClasses/{custom_class}``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

                This corresponds to the ``custom_class`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                The list of fields to be updated.
                This corresponds to the ``update_mask`` field
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
            google.cloud.speech_v1.types.CustomClass:
                A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [custom_class, update_mask]
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
        if not isinstance(request, cloud_speech_adaptation.UpdateCustomClassRequest):
            request = cloud_speech_adaptation.UpdateCustomClassRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if custom_class is not None:
            request.custom_class = custom_class
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_custom_class
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("custom_class.name", request.custom_class.name),)
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

    async def delete_custom_class(
        self,
        request: Optional[
            Union[cloud_speech_adaptation.DeleteCustomClassRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> None:
        r"""Delete a custom class.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1

            async def sample_delete_custom_class():
                # Create a client
                client = speech_v1.AdaptationAsyncClient()

                # Initialize request argument(s)
                request = speech_v1.DeleteCustomClassRequest(
                    name="name_value",
                )

                # Make the request
                await client.delete_custom_class(request=request)

        Args:
            request (Optional[Union[google.cloud.speech_v1.types.DeleteCustomClassRequest, dict]]):
                The request object. Message sent by the client for the ``DeleteCustomClass``
                method.
            name (:class:`str`):
                Required. The name of the custom class to delete.
                Format:

                ``projects/{project}/locations/{location}/customClasses/{custom_class}``

                Speech-to-Text supports three locations: ``global``,
                ``us`` (US North America), and ``eu`` (Europe). If you
                are calling the ``speech.googleapis.com`` endpoint, use
                the ``global`` location. To specify a region, use a
                `regional
                endpoint <https://cloud.google.com/speech-to-text/docs/endpoints>`__
                with matching ``us`` or ``eu`` location value.

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
        if not isinstance(request, cloud_speech_adaptation.DeleteCustomClassRequest):
            request = cloud_speech_adaptation.DeleteCustomClassRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_custom_class
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

    async def list_operations(
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
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors,
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
        rpc = self.transport._wrapped_methods[self._client._transport.list_operations]

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

    async def get_operation(
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
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors,
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
        rpc = self.transport._wrapped_methods[self._client._transport.get_operation]

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

    async def __aenter__(self) -> "AdaptationAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("AdaptationAsyncClient",)
