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
    AsyncIterable,
    AsyncIterator,
    Awaitable,
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

from google.cloud.speech_v1p1beta1 import gapic_version as package_version

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore

from google.cloud.speech_v1p1beta1.types import cloud_speech

from .client import SpeechClient
from .transports.base import DEFAULT_CLIENT_INFO, SpeechTransport
from .transports.grpc_asyncio import SpeechGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class SpeechAsyncClient:
    """Service that implements Google Cloud Speech API."""

    _client: SpeechClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = SpeechClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = SpeechClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = SpeechClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = SpeechClient._DEFAULT_UNIVERSE

    crypto_key_path = staticmethod(SpeechClient.crypto_key_path)
    parse_crypto_key_path = staticmethod(SpeechClient.parse_crypto_key_path)
    crypto_key_version_path = staticmethod(SpeechClient.crypto_key_version_path)
    parse_crypto_key_version_path = staticmethod(
        SpeechClient.parse_crypto_key_version_path
    )
    custom_class_path = staticmethod(SpeechClient.custom_class_path)
    parse_custom_class_path = staticmethod(SpeechClient.parse_custom_class_path)
    phrase_set_path = staticmethod(SpeechClient.phrase_set_path)
    parse_phrase_set_path = staticmethod(SpeechClient.parse_phrase_set_path)
    common_billing_account_path = staticmethod(SpeechClient.common_billing_account_path)
    parse_common_billing_account_path = staticmethod(
        SpeechClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(SpeechClient.common_folder_path)
    parse_common_folder_path = staticmethod(SpeechClient.parse_common_folder_path)
    common_organization_path = staticmethod(SpeechClient.common_organization_path)
    parse_common_organization_path = staticmethod(
        SpeechClient.parse_common_organization_path
    )
    common_project_path = staticmethod(SpeechClient.common_project_path)
    parse_common_project_path = staticmethod(SpeechClient.parse_common_project_path)
    common_location_path = staticmethod(SpeechClient.common_location_path)
    parse_common_location_path = staticmethod(SpeechClient.parse_common_location_path)

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            SpeechAsyncClient: The constructed client.
        """
        return SpeechClient.from_service_account_info.__func__(SpeechAsyncClient, info, *args, **kwargs)  # type: ignore

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
            SpeechAsyncClient: The constructed client.
        """
        return SpeechClient.from_service_account_file.__func__(SpeechAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return SpeechClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> SpeechTransport:
        """Returns the transport used by the client instance.

        Returns:
            SpeechTransport: The transport used by the client instance.
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

    get_transport_class = SpeechClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, SpeechTransport, Callable[..., SpeechTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the speech async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,SpeechTransport,Callable[..., SpeechTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the SpeechTransport constructor.
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
        self._client = SpeechClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.cloud.speech_v1p1beta1.SpeechAsyncClient`.",
                extra={
                    "serviceName": "google.cloud.speech.v1p1beta1.Speech",
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
                    "serviceName": "google.cloud.speech.v1p1beta1.Speech",
                    "credentialsType": None,
                },
            )

    async def recognize(
        self,
        request: Optional[Union[cloud_speech.RecognizeRequest, dict]] = None,
        *,
        config: Optional[cloud_speech.RecognitionConfig] = None,
        audio: Optional[cloud_speech.RecognitionAudio] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> cloud_speech.RecognizeResponse:
        r"""Performs synchronous speech recognition: receive
        results after all audio has been sent and processed.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1p1beta1

            async def sample_recognize():
                # Create a client
                client = speech_v1p1beta1.SpeechAsyncClient()

                # Initialize request argument(s)
                config = speech_v1p1beta1.RecognitionConfig()
                config.language_code = "language_code_value"

                audio = speech_v1p1beta1.RecognitionAudio()
                audio.content = b'content_blob'

                request = speech_v1p1beta1.RecognizeRequest(
                    config=config,
                    audio=audio,
                )

                # Make the request
                response = await client.recognize(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1p1beta1.types.RecognizeRequest, dict]]):
                The request object. The top-level message sent by the client for the
                ``Recognize`` method.
            config (:class:`google.cloud.speech_v1p1beta1.types.RecognitionConfig`):
                Required. Provides information to the
                recognizer that specifies how to process
                the request.

                This corresponds to the ``config`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            audio (:class:`google.cloud.speech_v1p1beta1.types.RecognitionAudio`):
                Required. The audio data to be
                recognized.

                This corresponds to the ``audio`` field
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
            google.cloud.speech_v1p1beta1.types.RecognizeResponse:
                The only message returned to the client by the Recognize method. It
                   contains the result as zero or more sequential
                   SpeechRecognitionResult messages.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [config, audio]
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
        if not isinstance(request, cloud_speech.RecognizeRequest):
            request = cloud_speech.RecognizeRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if config is not None:
            request.config = config
        if audio is not None:
            request.audio = audio

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.recognize
        ]

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

    async def long_running_recognize(
        self,
        request: Optional[Union[cloud_speech.LongRunningRecognizeRequest, dict]] = None,
        *,
        config: Optional[cloud_speech.RecognitionConfig] = None,
        audio: Optional[cloud_speech.RecognitionAudio] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Performs asynchronous speech recognition: receive results via
        the google.longrunning.Operations interface. Returns either an
        ``Operation.error`` or an ``Operation.response`` which contains
        a ``LongRunningRecognizeResponse`` message. For more information
        on asynchronous speech recognition, see the
        `how-to <https://cloud.google.com/speech-to-text/docs/async-recognize>`__.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1p1beta1

            async def sample_long_running_recognize():
                # Create a client
                client = speech_v1p1beta1.SpeechAsyncClient()

                # Initialize request argument(s)
                config = speech_v1p1beta1.RecognitionConfig()
                config.language_code = "language_code_value"

                audio = speech_v1p1beta1.RecognitionAudio()
                audio.content = b'content_blob'

                request = speech_v1p1beta1.LongRunningRecognizeRequest(
                    config=config,
                    audio=audio,
                )

                # Make the request
                operation = client.long_running_recognize(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.speech_v1p1beta1.types.LongRunningRecognizeRequest, dict]]):
                The request object. The top-level message sent by the client for the
                ``LongRunningRecognize`` method.
            config (:class:`google.cloud.speech_v1p1beta1.types.RecognitionConfig`):
                Required. Provides information to the
                recognizer that specifies how to process
                the request.

                This corresponds to the ``config`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            audio (:class:`google.cloud.speech_v1p1beta1.types.RecognitionAudio`):
                Required. The audio data to be
                recognized.

                This corresponds to the ``audio`` field
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
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.cloud.speech_v1p1beta1.types.LongRunningRecognizeResponse` The only message returned to the client by the LongRunningRecognize method.
                   It contains the result as zero or more sequential
                   SpeechRecognitionResult messages. It is included in
                   the result.response field of the Operation returned
                   by the GetOperation call of the
                   google::longrunning::Operations service.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [config, audio]
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
        if not isinstance(request, cloud_speech.LongRunningRecognizeRequest):
            request = cloud_speech.LongRunningRecognizeRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if config is not None:
            request.config = config
        if audio is not None:
            request.audio = audio

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.long_running_recognize
        ]

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            cloud_speech.LongRunningRecognizeResponse,
            metadata_type=cloud_speech.LongRunningRecognizeMetadata,
        )

        # Done; return the response.
        return response

    def streaming_recognize(
        self,
        requests: Optional[
            AsyncIterator[cloud_speech.StreamingRecognizeRequest]
        ] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> Awaitable[AsyncIterable[cloud_speech.StreamingRecognizeResponse]]:
        r"""Performs bidirectional streaming speech recognition:
        receive results while sending audio. This method is only
        available via the gRPC API (not REST).

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import speech_v1p1beta1

            async def sample_streaming_recognize():
                # Create a client
                client = speech_v1p1beta1.SpeechAsyncClient()

                # Initialize request argument(s)
                streaming_config = speech_v1p1beta1.StreamingRecognitionConfig()
                streaming_config.config.language_code = "language_code_value"

                request = speech_v1p1beta1.StreamingRecognizeRequest(
                    streaming_config=streaming_config,
                )

                # This method expects an iterator which contains
                # 'speech_v1p1beta1.StreamingRecognizeRequest' objects
                # Here we create a generator that yields a single `request` for
                # demonstrative purposes.
                requests = [request]

                def request_generator():
                    for request in requests:
                        yield request

                # Make the request
                stream = await client.streaming_recognize(requests=request_generator())

                # Handle the response
                async for response in stream:
                    print(response)

        Args:
            requests (AsyncIterator[`google.cloud.speech_v1p1beta1.types.StreamingRecognizeRequest`]):
                The request object AsyncIterator. The top-level message sent by the client for the
                ``StreamingRecognize`` method. Multiple
                ``StreamingRecognizeRequest`` messages are sent. The
                first message must contain a ``streaming_config``
                message and must not contain ``audio_content``. All
                subsequent messages must contain ``audio_content`` and
                must not contain a ``streaming_config`` message.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            AsyncIterable[google.cloud.speech_v1p1beta1.types.StreamingRecognizeResponse]:
                StreamingRecognizeResponse is the only message returned to the client by
                   StreamingRecognize. A series of zero or more
                   StreamingRecognizeResponse messages are streamed back
                   to the client. If there is no recognizable audio, and
                   single_utterance is set to false, then no messages
                   are streamed back to the client.

                   Here's an example of a series of
                   \`StreamingRecognizeResponse`s that might be returned
                   while processing audio:

                   1. results { alternatives { transcript: "tube" }
                      stability: 0.01 }

                   2. results { alternatives { transcript: "to be a" }
                      stability: 0.01 }

                   3. results { alternatives { transcript: "to be" }
                      stability: 0.9 } results { alternatives {
                      transcript: " or not to be" } stability: 0.01 }

                   4.

                      results { alternatives { transcript: "to be or not to be"
                         confidence: 0.92 } alternatives { transcript:
                         "to bee or not to bee" } is_final: true }

                   5. results { alternatives { transcript: " that's" }
                      stability: 0.01 }

                   6. results { alternatives { transcript: " that is" }
                      stability: 0.9 } results { alternatives {
                      transcript: " the question" } stability: 0.01 }

                   7.

                      results { alternatives { transcript: " that is the question"
                         confidence: 0.98 } alternatives { transcript: "
                         that was the question" } is_final: true }

                   Notes:

                   - Only two of the above responses #4 and #7 contain
                     final results; they are indicated by is_final:
                     true. Concatenating these together generates the
                     full transcript: "to be or not to be that is the
                     question".

                   - The others contain interim results. #3 and #6
                     contain two interim \`results\`: the first portion
                     has a high stability and is less likely to change;
                     the second portion has a low stability and is very
                     likely to change. A UI designer might choose to
                     show only high stability results.

                   - The specific stability and confidence values shown
                     above are only for illustrative purposes. Actual
                     values may vary.

                   -

                     In each response, only one of these fields will be set:
                        error, speech_event_type, or one or more
                        (repeated) results.

        """

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.streaming_recognize
        ]

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(
            requests,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

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

    async def __aenter__(self) -> "SpeechAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("SpeechAsyncClient",)
