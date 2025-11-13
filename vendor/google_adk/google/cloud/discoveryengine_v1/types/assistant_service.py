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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import assist_answer, search_service
from google.cloud.discoveryengine_v1.types import session as gcd_session

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "AssistUserMetadata",
        "StreamAssistRequest",
        "StreamAssistResponse",
    },
)


class AssistUserMetadata(proto.Message):
    r"""User metadata of the request.

    Attributes:
        time_zone (str):
            Optional. IANA time zone, e.g.
            Europe/Budapest.
        preferred_language_code (str):
            Optional. Preferred language to be used for
            answering if language detection fails. Also used
            as the language of error messages created by
            actions, regardless of language detection
            results.
    """

    time_zone: str = proto.Field(
        proto.STRING,
        number=1,
    )
    preferred_language_code: str = proto.Field(
        proto.STRING,
        number=2,
    )


class StreamAssistRequest(proto.Message):
    r"""Request for the
    [AssistantService.StreamAssist][google.cloud.discoveryengine.v1.AssistantService.StreamAssist]
    method.

    Attributes:
        name (str):
            Required. The resource name of the
            [Assistant][google.cloud.discoveryengine.v1.Assistant].
            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/assistants/{assistant}``
        query (google.cloud.discoveryengine_v1.types.Query):
            Optional. Current user query.

            Empty query is only supported if ``file_ids`` are provided.
            In this case, the answer will be generated based on those
            context files.
        session (str):
            Optional. The session to use for the request. If specified,
            the assistant has access to the session history, and the
            query and the answer are stored there.

            If ``-`` is specified as the session ID, or it is left
            empty, then a new session is created with an automatically
            generated ID.

            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/sessions/{session}``
        user_metadata (google.cloud.discoveryengine_v1.types.AssistUserMetadata):
            Optional. Information about the user
            initiating the query.
        tools_spec (google.cloud.discoveryengine_v1.types.StreamAssistRequest.ToolsSpec):
            Optional. Specification of tools that are
            used to serve the request.
        generation_spec (google.cloud.discoveryengine_v1.types.StreamAssistRequest.GenerationSpec):
            Optional. Specification of the generation
            configuration for the request.
    """

    class ToolsSpec(proto.Message):
        r"""Specification of tools that are used to serve the request.

        Attributes:
            vertex_ai_search_spec (google.cloud.discoveryengine_v1.types.StreamAssistRequest.ToolsSpec.VertexAiSearchSpec):
                Optional. Specification of the Vertex AI
                Search tool.
            web_grounding_spec (google.cloud.discoveryengine_v1.types.StreamAssistRequest.ToolsSpec.WebGroundingSpec):
                Optional. Specification of the web grounding tool. If field
                is present, enables grounding with web search. Works only if
                [Assistant.web_grounding_type][google.cloud.discoveryengine.v1.Assistant.web_grounding_type]
                is [WEB_GROUNDING_TYPE_GOOGLE_SEARCH][] or
                [WEB_GROUNDING_TYPE_ENTERPRISE_WEB_SEARCH][].
            image_generation_spec (google.cloud.discoveryengine_v1.types.StreamAssistRequest.ToolsSpec.ImageGenerationSpec):
                Optional. Specification of the image
                generation tool.
            video_generation_spec (google.cloud.discoveryengine_v1.types.StreamAssistRequest.ToolsSpec.VideoGenerationSpec):
                Optional. Specification of the video
                generation tool.
        """

        class VertexAiSearchSpec(proto.Message):
            r"""Specification of the Vertex AI Search tool.

            Attributes:
                data_store_specs (MutableSequence[google.cloud.discoveryengine_v1.types.SearchRequest.DataStoreSpec]):
                    Optional. Specs defining
                    [DataStore][google.cloud.discoveryengine.v1.DataStore]s to
                    filter on in a search call and configurations for those data
                    stores. This is only considered for
                    [Engine][google.cloud.discoveryengine.v1.Engine]s with
                    multiple data stores.
                filter (str):
                    Optional. The filter syntax consists of an expression
                    language for constructing a predicate from one or more
                    fields of the documents being filtered. Filter expression is
                    case-sensitive.

                    If this field is unrecognizable, an ``INVALID_ARGUMENT`` is
                    returned.

                    Filtering in Vertex AI Search is done by mapping the LHS
                    filter key to a key property defined in the Vertex AI Search
                    backend -- this mapping is defined by the customer in their
                    schema. For example a media customer might have a field
                    'name' in their schema. In this case the filter would look
                    like this: filter --> name:'ANY("king kong")'

                    For more information about filtering including syntax and
                    filter operators, see
                    `Filter <https://cloud.google.com/generative-ai-app-builder/docs/filter-search-metadata>`__
            """

            data_store_specs: MutableSequence[
                search_service.SearchRequest.DataStoreSpec
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=2,
                message=search_service.SearchRequest.DataStoreSpec,
            )
            filter: str = proto.Field(
                proto.STRING,
                number=4,
            )

        class WebGroundingSpec(proto.Message):
            r"""Specification of the web grounding tool."""

        class ImageGenerationSpec(proto.Message):
            r"""Specification of the image generation tool."""

        class VideoGenerationSpec(proto.Message):
            r"""Specification of the video generation tool."""

        vertex_ai_search_spec: "StreamAssistRequest.ToolsSpec.VertexAiSearchSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                message="StreamAssistRequest.ToolsSpec.VertexAiSearchSpec",
            )
        )
        web_grounding_spec: "StreamAssistRequest.ToolsSpec.WebGroundingSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=2,
                message="StreamAssistRequest.ToolsSpec.WebGroundingSpec",
            )
        )
        image_generation_spec: "StreamAssistRequest.ToolsSpec.ImageGenerationSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=3,
                message="StreamAssistRequest.ToolsSpec.ImageGenerationSpec",
            )
        )
        video_generation_spec: "StreamAssistRequest.ToolsSpec.VideoGenerationSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=4,
                message="StreamAssistRequest.ToolsSpec.VideoGenerationSpec",
            )
        )

    class GenerationSpec(proto.Message):
        r"""Assistant generation specification for the request.
        This allows to override the default generation configuration at
        the engine level.

        Attributes:
            model_id (str):
                Optional. The Vertex AI model_id used for the generative
                model. If not set, the default Assistant model will be used.
        """

        model_id: str = proto.Field(
            proto.STRING,
            number=1,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: gcd_session.Query = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_session.Query,
    )
    session: str = proto.Field(
        proto.STRING,
        number=3,
    )
    user_metadata: "AssistUserMetadata" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="AssistUserMetadata",
    )
    tools_spec: ToolsSpec = proto.Field(
        proto.MESSAGE,
        number=18,
        message=ToolsSpec,
    )
    generation_spec: GenerationSpec = proto.Field(
        proto.MESSAGE,
        number=19,
        message=GenerationSpec,
    )


class StreamAssistResponse(proto.Message):
    r"""Response for the
    [AssistantService.StreamAssist][google.cloud.discoveryengine.v1.AssistantService.StreamAssist]
    method.

    Attributes:
        answer (google.cloud.discoveryengine_v1.types.AssistAnswer):
            Assist answer resource object containing parts of the
            assistant's final answer for the user's query.

            Not present if the current response doesn't add anything to
            previously sent
            [AssistAnswer.replies][google.cloud.discoveryengine.v1.AssistAnswer.replies].

            Observe
            [AssistAnswer.state][google.cloud.discoveryengine.v1.AssistAnswer.state]
            to see if more parts are to be expected. While the state is
            ``IN_PROGRESS``, the
            [AssistAnswer.replies][google.cloud.discoveryengine.v1.AssistAnswer.replies]
            field in each response will contain replies (reply
            fragments) to be appended to the ones received in previous
            responses. [AssistAnswer.name][] won't be filled.

            If the state is ``SUCCEEDED``, ``FAILED`` or ``SKIPPED``,
            the response is the last response and [AssistAnswer.name][]
            will have a value.
        session_info (google.cloud.discoveryengine_v1.types.StreamAssistResponse.SessionInfo):
            Session information.
        assist_token (str):
            A global unique ID that identifies the
            current pair of request and stream of responses.
            Used for feedback and support.
    """

    class SessionInfo(proto.Message):
        r"""Information about the session.

        Attributes:
            session (str):
                Name of the newly generated or continued session.

                Format:
                ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/sessions/{session}``.
        """

        session: str = proto.Field(
            proto.STRING,
            number=1,
        )

    answer: assist_answer.AssistAnswer = proto.Field(
        proto.MESSAGE,
        number=1,
        message=assist_answer.AssistAnswer,
    )
    session_info: SessionInfo = proto.Field(
        proto.MESSAGE,
        number=2,
        message=SessionInfo,
    )
    assist_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
