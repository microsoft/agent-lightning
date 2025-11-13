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
import dataclasses
import json  # type: ignore
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1, rest_helpers, rest_streaming
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
import google.protobuf
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.discoveryengine_v1.types import conversation as gcd_conversation
from google.cloud.discoveryengine_v1.types import answer
from google.cloud.discoveryengine_v1.types import conversation
from google.cloud.discoveryengine_v1.types import conversational_search_service
from google.cloud.discoveryengine_v1.types import session
from google.cloud.discoveryengine_v1.types import session as gcd_session

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseConversationalSearchServiceRestTransport

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"requests@{requests_version}",
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class ConversationalSearchServiceRestInterceptor:
    """Interceptor for ConversationalSearchService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the ConversationalSearchServiceRestTransport.

    .. code-block:: python
        class MyCustomConversationalSearchServiceInterceptor(ConversationalSearchServiceRestInterceptor):
            def pre_answer_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_answer_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_converse_conversation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_converse_conversation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_conversation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_conversation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_session(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_conversation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_answer(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_answer(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_conversation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_conversation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_session(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_conversations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_conversations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_sessions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_sessions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_stream_answer_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_stream_answer_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_conversation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_conversation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_session(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = ConversationalSearchServiceRestTransport(interceptor=MyCustomConversationalSearchServiceInterceptor())
        client = ConversationalSearchServiceClient(transport=transport)


    """

    def pre_answer_query(
        self,
        request: conversational_search_service.AnswerQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.AnswerQueryRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for answer_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_answer_query(
        self, response: conversational_search_service.AnswerQueryResponse
    ) -> conversational_search_service.AnswerQueryResponse:
        """Post-rpc interceptor for answer_query

        DEPRECATED. Please use the `post_answer_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_answer_query` interceptor runs
        before the `post_answer_query_with_metadata` interceptor.
        """
        return response

    def post_answer_query_with_metadata(
        self,
        response: conversational_search_service.AnswerQueryResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.AnswerQueryResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for answer_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_answer_query_with_metadata`
        interceptor in new development instead of the `post_answer_query` interceptor.
        When both interceptors are used, this `post_answer_query_with_metadata` interceptor runs after the
        `post_answer_query` interceptor. The (possibly modified) response returned by
        `post_answer_query` will be passed to
        `post_answer_query_with_metadata`.
        """
        return response, metadata

    def pre_converse_conversation(
        self,
        request: conversational_search_service.ConverseConversationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.ConverseConversationRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for converse_conversation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_converse_conversation(
        self, response: conversational_search_service.ConverseConversationResponse
    ) -> conversational_search_service.ConverseConversationResponse:
        """Post-rpc interceptor for converse_conversation

        DEPRECATED. Please use the `post_converse_conversation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_converse_conversation` interceptor runs
        before the `post_converse_conversation_with_metadata` interceptor.
        """
        return response

    def post_converse_conversation_with_metadata(
        self,
        response: conversational_search_service.ConverseConversationResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.ConverseConversationResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for converse_conversation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_converse_conversation_with_metadata`
        interceptor in new development instead of the `post_converse_conversation` interceptor.
        When both interceptors are used, this `post_converse_conversation_with_metadata` interceptor runs after the
        `post_converse_conversation` interceptor. The (possibly modified) response returned by
        `post_converse_conversation` will be passed to
        `post_converse_conversation_with_metadata`.
        """
        return response, metadata

    def pre_create_conversation(
        self,
        request: conversational_search_service.CreateConversationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.CreateConversationRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_conversation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_create_conversation(
        self, response: gcd_conversation.Conversation
    ) -> gcd_conversation.Conversation:
        """Post-rpc interceptor for create_conversation

        DEPRECATED. Please use the `post_create_conversation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_create_conversation` interceptor runs
        before the `post_create_conversation_with_metadata` interceptor.
        """
        return response

    def post_create_conversation_with_metadata(
        self,
        response: gcd_conversation.Conversation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_conversation.Conversation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_conversation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_create_conversation_with_metadata`
        interceptor in new development instead of the `post_create_conversation` interceptor.
        When both interceptors are used, this `post_create_conversation_with_metadata` interceptor runs after the
        `post_create_conversation` interceptor. The (possibly modified) response returned by
        `post_create_conversation` will be passed to
        `post_create_conversation_with_metadata`.
        """
        return response, metadata

    def pre_create_session(
        self,
        request: conversational_search_service.CreateSessionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.CreateSessionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_create_session(self, response: gcd_session.Session) -> gcd_session.Session:
        """Post-rpc interceptor for create_session

        DEPRECATED. Please use the `post_create_session_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_create_session` interceptor runs
        before the `post_create_session_with_metadata` interceptor.
        """
        return response

    def post_create_session_with_metadata(
        self,
        response: gcd_session.Session,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_session.Session, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_session

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_create_session_with_metadata`
        interceptor in new development instead of the `post_create_session` interceptor.
        When both interceptors are used, this `post_create_session_with_metadata` interceptor runs after the
        `post_create_session` interceptor. The (possibly modified) response returned by
        `post_create_session` will be passed to
        `post_create_session_with_metadata`.
        """
        return response, metadata

    def pre_delete_conversation(
        self,
        request: conversational_search_service.DeleteConversationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.DeleteConversationRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_conversation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def pre_delete_session(
        self,
        request: conversational_search_service.DeleteSessionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.DeleteSessionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def pre_get_answer(
        self,
        request: conversational_search_service.GetAnswerRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.GetAnswerRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_answer

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_get_answer(self, response: answer.Answer) -> answer.Answer:
        """Post-rpc interceptor for get_answer

        DEPRECATED. Please use the `post_get_answer_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_get_answer` interceptor runs
        before the `post_get_answer_with_metadata` interceptor.
        """
        return response

    def post_get_answer_with_metadata(
        self, response: answer.Answer, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[answer.Answer, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_answer

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_get_answer_with_metadata`
        interceptor in new development instead of the `post_get_answer` interceptor.
        When both interceptors are used, this `post_get_answer_with_metadata` interceptor runs after the
        `post_get_answer` interceptor. The (possibly modified) response returned by
        `post_get_answer` will be passed to
        `post_get_answer_with_metadata`.
        """
        return response, metadata

    def pre_get_conversation(
        self,
        request: conversational_search_service.GetConversationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.GetConversationRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_conversation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_get_conversation(
        self, response: conversation.Conversation
    ) -> conversation.Conversation:
        """Post-rpc interceptor for get_conversation

        DEPRECATED. Please use the `post_get_conversation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_get_conversation` interceptor runs
        before the `post_get_conversation_with_metadata` interceptor.
        """
        return response

    def post_get_conversation_with_metadata(
        self,
        response: conversation.Conversation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[conversation.Conversation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_conversation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_get_conversation_with_metadata`
        interceptor in new development instead of the `post_get_conversation` interceptor.
        When both interceptors are used, this `post_get_conversation_with_metadata` interceptor runs after the
        `post_get_conversation` interceptor. The (possibly modified) response returned by
        `post_get_conversation` will be passed to
        `post_get_conversation_with_metadata`.
        """
        return response, metadata

    def pre_get_session(
        self,
        request: conversational_search_service.GetSessionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.GetSessionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_get_session(self, response: session.Session) -> session.Session:
        """Post-rpc interceptor for get_session

        DEPRECATED. Please use the `post_get_session_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_get_session` interceptor runs
        before the `post_get_session_with_metadata` interceptor.
        """
        return response

    def post_get_session_with_metadata(
        self,
        response: session.Session,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[session.Session, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_session

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_get_session_with_metadata`
        interceptor in new development instead of the `post_get_session` interceptor.
        When both interceptors are used, this `post_get_session_with_metadata` interceptor runs after the
        `post_get_session` interceptor. The (possibly modified) response returned by
        `post_get_session` will be passed to
        `post_get_session_with_metadata`.
        """
        return response, metadata

    def pre_list_conversations(
        self,
        request: conversational_search_service.ListConversationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.ListConversationsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_conversations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_list_conversations(
        self, response: conversational_search_service.ListConversationsResponse
    ) -> conversational_search_service.ListConversationsResponse:
        """Post-rpc interceptor for list_conversations

        DEPRECATED. Please use the `post_list_conversations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_list_conversations` interceptor runs
        before the `post_list_conversations_with_metadata` interceptor.
        """
        return response

    def post_list_conversations_with_metadata(
        self,
        response: conversational_search_service.ListConversationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.ListConversationsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_conversations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_list_conversations_with_metadata`
        interceptor in new development instead of the `post_list_conversations` interceptor.
        When both interceptors are used, this `post_list_conversations_with_metadata` interceptor runs after the
        `post_list_conversations` interceptor. The (possibly modified) response returned by
        `post_list_conversations` will be passed to
        `post_list_conversations_with_metadata`.
        """
        return response, metadata

    def pre_list_sessions(
        self,
        request: conversational_search_service.ListSessionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.ListSessionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_sessions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_list_sessions(
        self, response: conversational_search_service.ListSessionsResponse
    ) -> conversational_search_service.ListSessionsResponse:
        """Post-rpc interceptor for list_sessions

        DEPRECATED. Please use the `post_list_sessions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_list_sessions` interceptor runs
        before the `post_list_sessions_with_metadata` interceptor.
        """
        return response

    def post_list_sessions_with_metadata(
        self,
        response: conversational_search_service.ListSessionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.ListSessionsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_sessions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_list_sessions_with_metadata`
        interceptor in new development instead of the `post_list_sessions` interceptor.
        When both interceptors are used, this `post_list_sessions_with_metadata` interceptor runs after the
        `post_list_sessions` interceptor. The (possibly modified) response returned by
        `post_list_sessions` will be passed to
        `post_list_sessions_with_metadata`.
        """
        return response, metadata

    def pre_stream_answer_query(
        self,
        request: conversational_search_service.AnswerQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.AnswerQueryRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for stream_answer_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_stream_answer_query(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for stream_answer_query

        DEPRECATED. Please use the `post_stream_answer_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_stream_answer_query` interceptor runs
        before the `post_stream_answer_query_with_metadata` interceptor.
        """
        return response

    def post_stream_answer_query_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for stream_answer_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_stream_answer_query_with_metadata`
        interceptor in new development instead of the `post_stream_answer_query` interceptor.
        When both interceptors are used, this `post_stream_answer_query_with_metadata` interceptor runs after the
        `post_stream_answer_query` interceptor. The (possibly modified) response returned by
        `post_stream_answer_query` will be passed to
        `post_stream_answer_query_with_metadata`.
        """
        return response, metadata

    def pre_update_conversation(
        self,
        request: conversational_search_service.UpdateConversationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.UpdateConversationRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_conversation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_update_conversation(
        self, response: gcd_conversation.Conversation
    ) -> gcd_conversation.Conversation:
        """Post-rpc interceptor for update_conversation

        DEPRECATED. Please use the `post_update_conversation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_update_conversation` interceptor runs
        before the `post_update_conversation_with_metadata` interceptor.
        """
        return response

    def post_update_conversation_with_metadata(
        self,
        response: gcd_conversation.Conversation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_conversation.Conversation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_conversation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_update_conversation_with_metadata`
        interceptor in new development instead of the `post_update_conversation` interceptor.
        When both interceptors are used, this `post_update_conversation_with_metadata` interceptor runs after the
        `post_update_conversation` interceptor. The (possibly modified) response returned by
        `post_update_conversation` will be passed to
        `post_update_conversation_with_metadata`.
        """
        return response, metadata

    def pre_update_session(
        self,
        request: conversational_search_service.UpdateSessionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        conversational_search_service.UpdateSessionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_update_session(self, response: gcd_session.Session) -> gcd_session.Session:
        """Post-rpc interceptor for update_session

        DEPRECATED. Please use the `post_update_session_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code. This `post_update_session` interceptor runs
        before the `post_update_session_with_metadata` interceptor.
        """
        return response

    def post_update_session_with_metadata(
        self,
        response: gcd_session.Session,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_session.Session, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_session

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ConversationalSearchService server but before it is returned to user code.

        We recommend only using this `post_update_session_with_metadata`
        interceptor in new development instead of the `post_update_session` interceptor.
        When both interceptors are used, this `post_update_session_with_metadata` interceptor runs after the
        `post_update_session` interceptor. The (possibly modified) response returned by
        `post_update_session` will be passed to
        `post_update_session_with_metadata`.
        """
        return response, metadata

    def pre_cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.CancelOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code.
        """
        return response

    def pre_get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.GetOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code.
        """
        return response

    def pre_list_operations(
        self,
        request: operations_pb2.ListOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.ListOperationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConversationalSearchService server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the ConversationalSearchService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class ConversationalSearchServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: ConversationalSearchServiceRestInterceptor


class ConversationalSearchServiceRestTransport(
    _BaseConversationalSearchServiceRestTransport
):
    """REST backend synchronous transport for ConversationalSearchService.

    Service for conversational search.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "discoveryengine.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[ConversationalSearchServiceRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'discoveryengine.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            url_scheme=url_scheme,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or ConversationalSearchServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _AnswerQuery(
        _BaseConversationalSearchServiceRestTransport._BaseAnswerQuery,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.AnswerQuery")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.AnswerQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> conversational_search_service.AnswerQueryResponse:
            r"""Call the answer query method over HTTP.

            Args:
                request (~.conversational_search_service.AnswerQueryRequest):
                    The request object. Request message for
                [ConversationalSearchService.AnswerQuery][google.cloud.discoveryengine.v1.ConversationalSearchService.AnswerQuery]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.conversational_search_service.AnswerQueryResponse:
                    Response message for
                [ConversationalSearchService.AnswerQuery][google.cloud.discoveryengine.v1.ConversationalSearchService.AnswerQuery]
                method.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseAnswerQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_answer_query(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseAnswerQuery._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseAnswerQuery._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseAnswerQuery._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.AnswerQuery",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "AnswerQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._AnswerQuery._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = conversational_search_service.AnswerQueryResponse()
            pb_resp = conversational_search_service.AnswerQueryResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_answer_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_answer_query_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        conversational_search_service.AnswerQueryResponse.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.answer_query",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "AnswerQuery",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ConverseConversation(
        _BaseConversationalSearchServiceRestTransport._BaseConverseConversation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.ConverseConversation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.ConverseConversationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> conversational_search_service.ConverseConversationResponse:
            r"""Call the converse conversation method over HTTP.

            Args:
                request (~.conversational_search_service.ConverseConversationRequest):
                    The request object. Request message for
                [ConversationalSearchService.ConverseConversation][google.cloud.discoveryengine.v1.ConversationalSearchService.ConverseConversation]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.conversational_search_service.ConverseConversationResponse:
                    Response message for
                [ConversationalSearchService.ConverseConversation][google.cloud.discoveryengine.v1.ConversationalSearchService.ConverseConversation]
                method.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseConverseConversation._get_http_options()
            )

            request, metadata = self._interceptor.pre_converse_conversation(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseConverseConversation._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseConverseConversation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseConverseConversation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.ConverseConversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ConverseConversation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ConversationalSearchServiceRestTransport._ConverseConversation._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = conversational_search_service.ConverseConversationResponse()
            pb_resp = conversational_search_service.ConverseConversationResponse.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_converse_conversation(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_converse_conversation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = conversational_search_service.ConverseConversationResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.converse_conversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ConverseConversation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateConversation(
        _BaseConversationalSearchServiceRestTransport._BaseCreateConversation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.CreateConversation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.CreateConversationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_conversation.Conversation:
            r"""Call the create conversation method over HTTP.

            Args:
                request (~.conversational_search_service.CreateConversationRequest):
                    The request object. Request for CreateConversation
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_conversation.Conversation:
                    External conversation proto
                definition.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseCreateConversation._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_conversation(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseCreateConversation._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseCreateConversation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseCreateConversation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.CreateConversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "CreateConversation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ConversationalSearchServiceRestTransport._CreateConversation._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gcd_conversation.Conversation()
            pb_resp = gcd_conversation.Conversation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_conversation(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_conversation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_conversation.Conversation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.create_conversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "CreateConversation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateSession(
        _BaseConversationalSearchServiceRestTransport._BaseCreateSession,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.CreateSession")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.CreateSessionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_session.Session:
            r"""Call the create session method over HTTP.

            Args:
                request (~.conversational_search_service.CreateSessionRequest):
                    The request object. Request for CreateSession method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_session.Session:
                    External session proto definition.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseCreateSession._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_session(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseCreateSession._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseCreateSession._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseCreateSession._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.CreateSession",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "CreateSession",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._CreateSession._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gcd_session.Session()
            pb_resp = gcd_session.Session.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_session(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_session_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_session.Session.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.create_session",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "CreateSession",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteConversation(
        _BaseConversationalSearchServiceRestTransport._BaseDeleteConversation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.DeleteConversation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.DeleteConversationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete conversation method over HTTP.

            Args:
                request (~.conversational_search_service.DeleteConversationRequest):
                    The request object. Request for DeleteConversation
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseDeleteConversation._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_conversation(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseDeleteConversation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseDeleteConversation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.DeleteConversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "DeleteConversation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ConversationalSearchServiceRestTransport._DeleteConversation._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteSession(
        _BaseConversationalSearchServiceRestTransport._BaseDeleteSession,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.DeleteSession")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.DeleteSessionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete session method over HTTP.

            Args:
                request (~.conversational_search_service.DeleteSessionRequest):
                    The request object. Request for DeleteSession method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseDeleteSession._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_session(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseDeleteSession._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseDeleteSession._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.DeleteSession",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "DeleteSession",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._DeleteSession._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GetAnswer(
        _BaseConversationalSearchServiceRestTransport._BaseGetAnswer,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.GetAnswer")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.GetAnswerRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> answer.Answer:
            r"""Call the get answer method over HTTP.

            Args:
                request (~.conversational_search_service.GetAnswerRequest):
                    The request object. Request for GetAnswer method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.answer.Answer:
                    Defines an answer.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseGetAnswer._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_answer(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseGetAnswer._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseGetAnswer._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.GetAnswer",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetAnswer",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._GetAnswer._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = answer.Answer()
            pb_resp = answer.Answer.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_answer(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_answer_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = answer.Answer.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.get_answer",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetAnswer",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetConversation(
        _BaseConversationalSearchServiceRestTransport._BaseGetConversation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.GetConversation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.GetConversationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> conversation.Conversation:
            r"""Call the get conversation method over HTTP.

            Args:
                request (~.conversational_search_service.GetConversationRequest):
                    The request object. Request for GetConversation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.conversation.Conversation:
                    External conversation proto
                definition.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseGetConversation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_conversation(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseGetConversation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseGetConversation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.GetConversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetConversation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._GetConversation._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = conversation.Conversation()
            pb_resp = conversation.Conversation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_conversation(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_conversation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = conversation.Conversation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.get_conversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetConversation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetSession(
        _BaseConversationalSearchServiceRestTransport._BaseGetSession,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.GetSession")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.GetSessionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> session.Session:
            r"""Call the get session method over HTTP.

            Args:
                request (~.conversational_search_service.GetSessionRequest):
                    The request object. Request for GetSession method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.session.Session:
                    External session proto definition.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseGetSession._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_session(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseGetSession._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseGetSession._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.GetSession",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetSession",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._GetSession._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = session.Session()
            pb_resp = session.Session.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_session(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_session_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = session.Session.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.get_session",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetSession",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListConversations(
        _BaseConversationalSearchServiceRestTransport._BaseListConversations,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.ListConversations")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.ListConversationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> conversational_search_service.ListConversationsResponse:
            r"""Call the list conversations method over HTTP.

            Args:
                request (~.conversational_search_service.ListConversationsRequest):
                    The request object. Request for ListConversations method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.conversational_search_service.ListConversationsResponse:
                    Response for ListConversations
                method.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseListConversations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_conversations(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseListConversations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseListConversations._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.ListConversations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ListConversations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ConversationalSearchServiceRestTransport._ListConversations._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = conversational_search_service.ListConversationsResponse()
            pb_resp = conversational_search_service.ListConversationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_conversations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_conversations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        conversational_search_service.ListConversationsResponse.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.list_conversations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ListConversations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListSessions(
        _BaseConversationalSearchServiceRestTransport._BaseListSessions,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.ListSessions")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.ListSessionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> conversational_search_service.ListSessionsResponse:
            r"""Call the list sessions method over HTTP.

            Args:
                request (~.conversational_search_service.ListSessionsRequest):
                    The request object. Request for ListSessions method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.conversational_search_service.ListSessionsResponse:
                    Response for ListSessions method.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseListSessions._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_sessions(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseListSessions._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseListSessions._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.ListSessions",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ListSessions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._ListSessions._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = conversational_search_service.ListSessionsResponse()
            pb_resp = conversational_search_service.ListSessionsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_sessions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_sessions_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        conversational_search_service.ListSessionsResponse.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.list_sessions",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ListSessions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _StreamAnswerQuery(
        _BaseConversationalSearchServiceRestTransport._BaseStreamAnswerQuery,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.StreamAnswerQuery")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                stream=True,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.AnswerQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the stream answer query method over HTTP.

            Args:
                request (~.conversational_search_service.AnswerQueryRequest):
                    The request object. Request message for
                [ConversationalSearchService.AnswerQuery][google.cloud.discoveryengine.v1.ConversationalSearchService.AnswerQuery]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.conversational_search_service.AnswerQueryResponse:
                    Response message for
                [ConversationalSearchService.AnswerQuery][google.cloud.discoveryengine.v1.ConversationalSearchService.AnswerQuery]
                method.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseStreamAnswerQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_stream_answer_query(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseStreamAnswerQuery._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseStreamAnswerQuery._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseStreamAnswerQuery._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.StreamAnswerQuery",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "StreamAnswerQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ConversationalSearchServiceRestTransport._StreamAnswerQuery._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(
                response, conversational_search_service.AnswerQueryResponse
            )

            resp = self._interceptor.post_stream_answer_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_stream_answer_query_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                http_response = {
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.stream_answer_query",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "StreamAnswerQuery",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateConversation(
        _BaseConversationalSearchServiceRestTransport._BaseUpdateConversation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.UpdateConversation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.UpdateConversationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_conversation.Conversation:
            r"""Call the update conversation method over HTTP.

            Args:
                request (~.conversational_search_service.UpdateConversationRequest):
                    The request object. Request for UpdateConversation
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_conversation.Conversation:
                    External conversation proto
                definition.

            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseUpdateConversation._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_conversation(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseUpdateConversation._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseUpdateConversation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseUpdateConversation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.UpdateConversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "UpdateConversation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ConversationalSearchServiceRestTransport._UpdateConversation._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gcd_conversation.Conversation()
            pb_resp = gcd_conversation.Conversation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_conversation(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_conversation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_conversation.Conversation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.update_conversation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "UpdateConversation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateSession(
        _BaseConversationalSearchServiceRestTransport._BaseUpdateSession,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.UpdateSession")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: conversational_search_service.UpdateSessionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_session.Session:
            r"""Call the update session method over HTTP.

            Args:
                request (~.conversational_search_service.UpdateSessionRequest):
                    The request object. Request for UpdateSession method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_session.Session:
                    External session proto definition.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseUpdateSession._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_session(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseUpdateSession._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseUpdateSession._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseUpdateSession._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.UpdateSession",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "UpdateSession",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._UpdateSession._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = gcd_session.Session()
            pb_resp = gcd_session.Session.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_session(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_session_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_session.Session.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.update_session",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "UpdateSession",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def answer_query(
        self,
    ) -> Callable[
        [conversational_search_service.AnswerQueryRequest],
        conversational_search_service.AnswerQueryResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._AnswerQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def converse_conversation(
        self,
    ) -> Callable[
        [conversational_search_service.ConverseConversationRequest],
        conversational_search_service.ConverseConversationResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ConverseConversation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_conversation(
        self,
    ) -> Callable[
        [conversational_search_service.CreateConversationRequest],
        gcd_conversation.Conversation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateConversation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_session(
        self,
    ) -> Callable[
        [conversational_search_service.CreateSessionRequest], gcd_session.Session
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateSession(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_conversation(
        self,
    ) -> Callable[
        [conversational_search_service.DeleteConversationRequest], empty_pb2.Empty
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteConversation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_session(
        self,
    ) -> Callable[
        [conversational_search_service.DeleteSessionRequest], empty_pb2.Empty
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteSession(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_answer(
        self,
    ) -> Callable[[conversational_search_service.GetAnswerRequest], answer.Answer]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetAnswer(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_conversation(
        self,
    ) -> Callable[
        [conversational_search_service.GetConversationRequest],
        conversation.Conversation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetConversation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_session(
        self,
    ) -> Callable[[conversational_search_service.GetSessionRequest], session.Session]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetSession(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_conversations(
        self,
    ) -> Callable[
        [conversational_search_service.ListConversationsRequest],
        conversational_search_service.ListConversationsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListConversations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_sessions(
        self,
    ) -> Callable[
        [conversational_search_service.ListSessionsRequest],
        conversational_search_service.ListSessionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListSessions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def stream_answer_query(
        self,
    ) -> Callable[
        [conversational_search_service.AnswerQueryRequest],
        conversational_search_service.AnswerQueryResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._StreamAnswerQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_conversation(
        self,
    ) -> Callable[
        [conversational_search_service.UpdateConversationRequest],
        gcd_conversation.Conversation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateConversation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_session(
        self,
    ) -> Callable[
        [conversational_search_service.UpdateSessionRequest], gcd_session.Session
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateSession(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseConversationalSearchServiceRestTransport._BaseCancelOperation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.CancelOperation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: operations_pb2.CancelOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> None:
            r"""Call the cancel operation method over HTTP.

            Args:
                request (operations_pb2.CancelOperationRequest):
                    The request object for CancelOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            body = _BaseConversationalSearchServiceRestTransport._BaseCancelOperation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseCancelOperation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._CancelOperation._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            return self._interceptor.post_cancel_operation(None)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseConversationalSearchServiceRestTransport._BaseGetOperation,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.GetOperation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: operations_pb2.GetOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the get operation method over HTTP.

            Args:
                request (operations_pb2.GetOperationRequest):
                    The request object for GetOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.Operation: Response from GetOperation method.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseGetOperation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._GetOperation._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            content = response.content.decode("utf-8")
            resp = operations_pb2.Operation()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_get_operation(resp)
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "GetOperation",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def list_operations(self):
        return self._ListOperations(self._session, self._host, self._interceptor)  # type: ignore

    class _ListOperations(
        _BaseConversationalSearchServiceRestTransport._BaseListOperations,
        ConversationalSearchServiceRestStub,
    ):
        def __hash__(self):
            return hash("ConversationalSearchServiceRestTransport.ListOperations")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: operations_pb2.ListOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.ListOperationsResponse:
            r"""Call the list operations method over HTTP.

            Args:
                request (operations_pb2.ListOperationsRequest):
                    The request object for ListOperations method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.ListOperationsResponse: Response from ListOperations method.
            """

            http_options = (
                _BaseConversationalSearchServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseConversationalSearchServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseConversationalSearchServiceRestTransport._BaseListOperations._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1.ConversationalSearchServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ConversationalSearchServiceRestTransport._ListOperations._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            content = response.content.decode("utf-8")
            resp = operations_pb2.ListOperationsResponse()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_list_operations(resp)
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1.ConversationalSearchServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1.ConversationalSearchService",
                        "rpcName": "ListOperations",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("ConversationalSearchServiceRestTransport",)
