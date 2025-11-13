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

from google.cloud.discoveryengine_v1.types import grounding

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "GroundedGenerationContent",
        "GenerateGroundedContentRequest",
        "GenerateGroundedContentResponse",
        "CheckGroundingSpec",
        "CheckGroundingRequest",
        "CheckGroundingResponse",
    },
)


class GroundedGenerationContent(proto.Message):
    r"""Base structured datatype containing multi-part content of a
    message.

    Attributes:
        role (str):
            Producer of the content. Must be either ``user`` or
            ``model``.

            Intended to be used for multi-turn conversations. Otherwise,
            it can be left unset.
        parts (MutableSequence[google.cloud.discoveryengine_v1.types.GroundedGenerationContent.Part]):
            Ordered ``Parts`` that constitute a single message.
    """

    class Part(proto.Message):
        r"""Single part of content.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            text (str):
                Inline text.

                This field is a member of `oneof`_ ``data``.
        """

        text: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="data",
        )

    role: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parts: MutableSequence[Part] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=Part,
    )


class GenerateGroundedContentRequest(proto.Message):
    r"""Top-level message sent by the client for the
    ``GenerateGroundedContent`` method.

    Attributes:
        location (str):
            Required. Location resource.

            Format: ``projects/{project}/locations/{location}``.
        system_instruction (google.cloud.discoveryengine_v1.types.GroundedGenerationContent):
            Content of the system instruction for the
            current API.
            These instructions will take priority over any
            other prompt instructions if the selected model
            is supporting them.
        contents (MutableSequence[google.cloud.discoveryengine_v1.types.GroundedGenerationContent]):
            Content of the current conversation with the
            model.
            For single-turn queries, this is a single
            instance. For multi-turn queries, this is a
            repeated field that contains conversation
            history + latest request.
        generation_spec (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GenerationSpec):
            Content generation specification.
        grounding_spec (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GroundingSpec):
            Grounding specification.
        user_labels (MutableMapping[str, str]):
            The user labels applied to a resource must meet the
            following requirements:

            - Each resource can have multiple labels, up to a maximum of
              64.
            - Each label must be a key-value pair.
            - Keys have a minimum length of 1 character and a maximum
              length of 63 characters and cannot be empty. Values can be
              empty and have a maximum length of 63 characters.
            - Keys and values can contain only lowercase letters,
              numeric characters, underscores, and dashes. All
              characters must use UTF-8 encoding, and international
              characters are allowed.
            - The key portion of a label must be unique. However, you
              can use the same key with multiple resources.
            - Keys must start with a lowercase letter or international
              character.

            See `Google Cloud
            Document <https://cloud.google.com/resource-manager/docs/creating-managing-labels#requirements>`__
            for more details.
    """

    class GenerationSpec(proto.Message):
        r"""Content generation specification.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            model_id (str):
                Specifies which Vertex model id to use for
                generation.
            language_code (str):
                Language code for content. Use language tags defined by
                `BCP47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__.
            temperature (float):
                If specified, custom value for the
                temperature will be used.

                This field is a member of `oneof`_ ``_temperature``.
            top_p (float):
                If specified, custom value for nucleus
                sampling will be used.

                This field is a member of `oneof`_ ``_top_p``.
            top_k (int):
                If specified, custom value for top-k sampling
                will be used.

                This field is a member of `oneof`_ ``_top_k``.
            frequency_penalty (float):
                If specified, custom value for frequency
                penalty will be used.

                This field is a member of `oneof`_ ``_frequency_penalty``.
            seed (int):
                If specified, custom value for the seed will
                be used.

                This field is a member of `oneof`_ ``_seed``.
            presence_penalty (float):
                If specified, custom value for presence
                penalty will be used.

                This field is a member of `oneof`_ ``_presence_penalty``.
            max_output_tokens (int):
                If specified, custom value for max output
                tokens will be used.

                This field is a member of `oneof`_ ``_max_output_tokens``.
        """

        model_id: str = proto.Field(
            proto.STRING,
            number=3,
        )
        language_code: str = proto.Field(
            proto.STRING,
            number=2,
        )
        temperature: float = proto.Field(
            proto.FLOAT,
            number=4,
            optional=True,
        )
        top_p: float = proto.Field(
            proto.FLOAT,
            number=5,
            optional=True,
        )
        top_k: int = proto.Field(
            proto.INT32,
            number=7,
            optional=True,
        )
        frequency_penalty: float = proto.Field(
            proto.FLOAT,
            number=8,
            optional=True,
        )
        seed: int = proto.Field(
            proto.INT32,
            number=12,
            optional=True,
        )
        presence_penalty: float = proto.Field(
            proto.FLOAT,
            number=9,
            optional=True,
        )
        max_output_tokens: int = proto.Field(
            proto.INT32,
            number=10,
            optional=True,
        )

    class DynamicRetrievalConfiguration(proto.Message):
        r"""Describes the options to customize dynamic retrieval.

        Attributes:
            predictor (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.DynamicRetrievalConfiguration.DynamicRetrievalPredictor):
                Specification for the predictor for dynamic
                retrieval.
        """

        class DynamicRetrievalPredictor(proto.Message):
            r"""Describes the predictor settings for dynamic retrieval.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                version (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.DynamicRetrievalConfiguration.DynamicRetrievalPredictor.Version):
                    The version of the predictor to be used in
                    dynamic retrieval.
                threshold (float):
                    The value of the threshold. If the predictor
                    will predict a value smaller than this, it would
                    suppress grounding in the source.

                    This field is a member of `oneof`_ ``_threshold``.
            """

            class Version(proto.Enum):
                r"""The version of the predictor to be used in dynamic retrieval.

                Values:
                    VERSION_UNSPECIFIED (0):
                        Automatically choose the best version of the
                        retrieval predictor.
                    V1_INDEPENDENT (1):
                        The V1 model which is evaluating each source
                        independently.
                """
                VERSION_UNSPECIFIED = 0
                V1_INDEPENDENT = 1

            version: "GenerateGroundedContentRequest.DynamicRetrievalConfiguration.DynamicRetrievalPredictor.Version" = proto.Field(
                proto.ENUM,
                number=1,
                enum="GenerateGroundedContentRequest.DynamicRetrievalConfiguration.DynamicRetrievalPredictor.Version",
            )
            threshold: float = proto.Field(
                proto.FLOAT,
                number=2,
                optional=True,
            )

        predictor: "GenerateGroundedContentRequest.DynamicRetrievalConfiguration.DynamicRetrievalPredictor" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="GenerateGroundedContentRequest.DynamicRetrievalConfiguration.DynamicRetrievalPredictor",
        )

    class GroundingSource(proto.Message):
        r"""Grounding source.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            inline_source (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GroundingSource.InlineSource):
                If set, grounding is performed with inline
                content.

                This field is a member of `oneof`_ ``source``.
            search_source (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GroundingSource.SearchSource):
                If set, grounding is performed with Vertex AI
                Search.

                This field is a member of `oneof`_ ``source``.
            google_search_source (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GroundingSource.GoogleSearchSource):
                If set, grounding is performed with Google
                Search.

                This field is a member of `oneof`_ ``source``.
            enterprise_web_retrieval_source (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GroundingSource.EnterpriseWebRetrievalSource):
                If set, grounding is performed with
                enterprise web retrieval.

                This field is a member of `oneof`_ ``source``.
        """

        class InlineSource(proto.Message):
            r"""Message to be used for grounding based on inline content.

            Attributes:
                grounding_facts (MutableSequence[google.cloud.discoveryengine_v1.types.GroundingFact]):
                    List of facts to be used for grounding.
                attributes (MutableMapping[str, str]):
                    Attributes associated with the content.

                    Common attributes include ``source`` (indicating where the
                    content was sourced from) and ``author`` (indicating the
                    author of the content).
            """

            grounding_facts: MutableSequence[
                grounding.GroundingFact
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message=grounding.GroundingFact,
            )
            attributes: MutableMapping[str, str] = proto.MapField(
                proto.STRING,
                proto.STRING,
                number=2,
            )

        class SearchSource(proto.Message):
            r"""Message to be used for grounding with Vertex AI Search.

            Attributes:
                serving_config (str):
                    The resource name of the Engine to use.

                    Format:
                    ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}/servingConfigs/{serving_config_id}``
                max_result_count (int):
                    Number of search results to return.

                    The default value is 10. The maximumm allowed
                    value is 10.
                filter (str):
                    Filter expression to be applied to the search.

                    The syntax is the same as
                    [SearchRequest.filter][google.cloud.discoveryengine.v1.SearchRequest.filter].
                safe_search (bool):
                    If set, safe search is enabled in Vertex AI
                    Search requests.
            """

            serving_config: str = proto.Field(
                proto.STRING,
                number=1,
            )
            max_result_count: int = proto.Field(
                proto.INT32,
                number=2,
            )
            filter: str = proto.Field(
                proto.STRING,
                number=3,
            )
            safe_search: bool = proto.Field(
                proto.BOOL,
                number=5,
            )

        class GoogleSearchSource(proto.Message):
            r"""Google Search config parameters.

            Attributes:
                dynamic_retrieval_config (google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.DynamicRetrievalConfiguration):
                    Optional. Specifies the dynamic retrieval
                    configuration for the given source.
            """

            dynamic_retrieval_config: "GenerateGroundedContentRequest.DynamicRetrievalConfiguration" = proto.Field(
                proto.MESSAGE,
                number=2,
                message="GenerateGroundedContentRequest.DynamicRetrievalConfiguration",
            )

        class EnterpriseWebRetrievalSource(proto.Message):
            r"""Params for using enterprise web retrieval as grounding
            source.

            """

        inline_source: "GenerateGroundedContentRequest.GroundingSource.InlineSource" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="source",
                message="GenerateGroundedContentRequest.GroundingSource.InlineSource",
            )
        )
        search_source: "GenerateGroundedContentRequest.GroundingSource.SearchSource" = (
            proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="source",
                message="GenerateGroundedContentRequest.GroundingSource.SearchSource",
            )
        )
        google_search_source: "GenerateGroundedContentRequest.GroundingSource.GoogleSearchSource" = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="source",
            message="GenerateGroundedContentRequest.GroundingSource.GoogleSearchSource",
        )
        enterprise_web_retrieval_source: "GenerateGroundedContentRequest.GroundingSource.EnterpriseWebRetrievalSource" = proto.Field(
            proto.MESSAGE,
            number=8,
            oneof="source",
            message="GenerateGroundedContentRequest.GroundingSource.EnterpriseWebRetrievalSource",
        )

    class GroundingSpec(proto.Message):
        r"""Grounding specification.

        Attributes:
            grounding_sources (MutableSequence[google.cloud.discoveryengine_v1.types.GenerateGroundedContentRequest.GroundingSource]):
                Grounding sources.
        """

        grounding_sources: MutableSequence[
            "GenerateGroundedContentRequest.GroundingSource"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="GenerateGroundedContentRequest.GroundingSource",
        )

    location: str = proto.Field(
        proto.STRING,
        number=1,
    )
    system_instruction: "GroundedGenerationContent" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="GroundedGenerationContent",
    )
    contents: MutableSequence["GroundedGenerationContent"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="GroundedGenerationContent",
    )
    generation_spec: GenerationSpec = proto.Field(
        proto.MESSAGE,
        number=3,
        message=GenerationSpec,
    )
    grounding_spec: GroundingSpec = proto.Field(
        proto.MESSAGE,
        number=4,
        message=GroundingSpec,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=6,
    )


class GenerateGroundedContentResponse(proto.Message):
    r"""Response for the ``GenerateGroundedContent`` method.

    Attributes:
        candidates (MutableSequence[google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate]):
            Generated candidates.
    """

    class Candidate(proto.Message):
        r"""A response candidate generated from the model.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            index (int):
                Index of the candidate.
            content (google.cloud.discoveryengine_v1.types.GroundedGenerationContent):
                Content of the candidate.
            grounding_score (float):
                The overall grounding score for the candidate, in the range
                of [0, 1].

                This field is a member of `oneof`_ ``_grounding_score``.
            grounding_metadata (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata):
                Grounding metadata for the generated content.
        """

        class GroundingMetadata(proto.Message):
            r"""Citation for the generated content.

            Attributes:
                retrieval_metadata (MutableSequence[google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.RetrievalMetadata]):
                    Retrieval metadata to provide an
                    understanding in the retrieval steps performed
                    by the model. There can be multiple such
                    messages which can correspond to different parts
                    of the retrieval. This is a mechanism used to
                    ensure transparency to our users.
                support_chunks (MutableSequence[google.cloud.discoveryengine_v1.types.FactChunk]):
                    List of chunks to be attributed across all
                    claims in the candidate. These are derived from
                    the grounding sources supplied in the request.
                web_search_queries (MutableSequence[str]):
                    Web search queries for the following-up web
                    search.
                search_entry_point (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.SearchEntryPoint):
                    Google search entry for the following-up web
                    searches.
                grounding_support (MutableSequence[google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.GroundingSupport]):
                    GroundingSupport across all claims in the
                    answer candidate. An support to a fact indicates
                    that the claim is supported by the fact.
                images (MutableSequence[google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata]):
                    Images from the web search.
            """

            class RetrievalMetadata(proto.Message):
                r"""Describes the metadata associated with a retrieval step.

                Attributes:
                    source (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.RetrievalMetadata.Source):
                        Describes the source to which the metadata is
                        referring to.
                    dynamic_retrieval_metadata (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalMetadata):
                        Metadata for dynamic retrieval.
                """

                class Source(proto.Enum):
                    r"""Describes the source to which the metadata is associated to.

                    Values:
                        SOURCE_UNSPECIFIED (0):
                            Unspecified source.
                        VERTEX_AI_SEARCH (1):
                            Vertex AI search.
                        GOOGLE_SEARCH (3):
                            Google Search.
                        INLINE_CONTENT (2):
                            User inline provided content.
                        GOOGLE_MAPS (4):
                            Google Maps.
                    """
                    SOURCE_UNSPECIFIED = 0
                    VERTEX_AI_SEARCH = 1
                    GOOGLE_SEARCH = 3
                    INLINE_CONTENT = 2
                    GOOGLE_MAPS = 4

                source: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.RetrievalMetadata.Source" = proto.Field(
                    proto.ENUM,
                    number=1,
                    enum="GenerateGroundedContentResponse.Candidate.GroundingMetadata.RetrievalMetadata.Source",
                )
                dynamic_retrieval_metadata: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalMetadata" = proto.Field(
                    proto.MESSAGE,
                    number=2,
                    message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalMetadata",
                )

            class DynamicRetrievalMetadata(proto.Message):
                r"""Describes the metadata about dynamic retrieval.

                Attributes:
                    predictor_metadata (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalPredictorMetadata):
                        Metadata for the dynamic retrieval predictor.
                """

                predictor_metadata: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalPredictorMetadata" = proto.Field(
                    proto.MESSAGE,
                    number=1,
                    message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalPredictorMetadata",
                )

            class DynamicRetrievalPredictorMetadata(proto.Message):
                r"""Describes the metadata about the dynamic retrieval predictor.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    version (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalPredictorMetadata.Version):
                        The version of the predictor which was used
                        in dynamic retrieval.
                    prediction (float):
                        The value of the predictor. This should be between [0, 1]
                        where a value of 0 means that the query would not benefit
                        from grounding, while a value of 1.0 means that the query
                        would benefit the most. In between values allow to
                        differentiate between different usefulness scores for
                        grounding.

                        This field is a member of `oneof`_ ``_prediction``.
                """

                class Version(proto.Enum):
                    r"""The version of the predictor which was used in dynamic
                    retrieval.

                    Values:
                        VERSION_UNSPECIFIED (0):
                            Unspecified version, should never be used.
                        V1_INDEPENDENT (1):
                            The V1 model which is evaluating each source
                            independently.
                    """
                    VERSION_UNSPECIFIED = 0
                    V1_INDEPENDENT = 1

                version: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalPredictorMetadata.Version" = proto.Field(
                    proto.ENUM,
                    number=1,
                    enum="GenerateGroundedContentResponse.Candidate.GroundingMetadata.DynamicRetrievalPredictorMetadata.Version",
                )
                prediction: float = proto.Field(
                    proto.FLOAT,
                    number=2,
                    optional=True,
                )

            class SearchEntryPoint(proto.Message):
                r"""Google search entry point.

                Attributes:
                    rendered_content (str):
                        Web content snippet that can be embedded in a
                        web page or an app webview.
                    sdk_blob (bytes):
                        Base64 encoded JSON representing array of
                        <search term, search url> tuple.
                """

                rendered_content: str = proto.Field(
                    proto.STRING,
                    number=1,
                )
                sdk_blob: bytes = proto.Field(
                    proto.BYTES,
                    number=2,
                )

            class GroundingSupport(proto.Message):
                r"""Grounding info for a claim in the candidate and its support.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    claim_text (str):
                        Text for the claim in the candidate. Always
                        provided when a support is found.
                    support_chunk_indices (MutableSequence[int]):
                        A list of indices (into 'support_chunks') specifying the
                        citations associated with the claim. For instance [1,3,4]
                        means that support_chunks[1], support_chunks[3],
                        support_chunks[4] are the chunks attributed to the claim.
                    support_score (float):
                        A score in the range of [0, 1] describing how grounded is a
                        specific claim in the support chunks indicated. Higher value
                        means that the claim is better supported by the chunks.

                        This field is a member of `oneof`_ ``_support_score``.
                """

                claim_text: str = proto.Field(
                    proto.STRING,
                    number=1,
                )
                support_chunk_indices: MutableSequence[int] = proto.RepeatedField(
                    proto.INT32,
                    number=3,
                )
                support_score: float = proto.Field(
                    proto.FLOAT,
                    number=2,
                    optional=True,
                )

            class ImageMetadata(proto.Message):
                r"""Metadata about an image from the web search.

                Attributes:
                    image (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.Image):
                        Metadata about the full size image.
                    thumbnail (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.Image):
                        Metadata about the thumbnail.
                    source (google.cloud.discoveryengine_v1.types.GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.WebsiteInfo):
                        The details about the website that the image
                        is from.
                """

                class WebsiteInfo(proto.Message):
                    r"""Metadata about the website that the image is from.

                    Attributes:
                        uri (str):
                            The url of the website.
                        title (str):
                            The title of the website.
                    """

                    uri: str = proto.Field(
                        proto.STRING,
                        number=1,
                    )
                    title: str = proto.Field(
                        proto.STRING,
                        number=2,
                    )

                class Image(proto.Message):
                    r"""Metadata about the image.

                    Attributes:
                        uri (str):
                            The url of the image.
                        width (int):
                            The width of the image in pixels.
                        height (int):
                            The height of the image in pixels.
                    """

                    uri: str = proto.Field(
                        proto.STRING,
                        number=1,
                    )
                    width: int = proto.Field(
                        proto.INT32,
                        number=2,
                    )
                    height: int = proto.Field(
                        proto.INT32,
                        number=3,
                    )

                image: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.Image" = proto.Field(
                    proto.MESSAGE,
                    number=1,
                    message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.Image",
                )
                thumbnail: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.Image" = proto.Field(
                    proto.MESSAGE,
                    number=2,
                    message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.Image",
                )
                source: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.WebsiteInfo" = proto.Field(
                    proto.MESSAGE,
                    number=3,
                    message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata.WebsiteInfo",
                )

            retrieval_metadata: MutableSequence[
                "GenerateGroundedContentResponse.Candidate.GroundingMetadata.RetrievalMetadata"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=5,
                message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.RetrievalMetadata",
            )
            support_chunks: MutableSequence[grounding.FactChunk] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message=grounding.FactChunk,
            )
            web_search_queries: MutableSequence[str] = proto.RepeatedField(
                proto.STRING,
                number=3,
            )
            search_entry_point: "GenerateGroundedContentResponse.Candidate.GroundingMetadata.SearchEntryPoint" = proto.Field(
                proto.MESSAGE,
                number=4,
                message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.SearchEntryPoint",
            )
            grounding_support: MutableSequence[
                "GenerateGroundedContentResponse.Candidate.GroundingMetadata.GroundingSupport"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=2,
                message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.GroundingSupport",
            )
            images: MutableSequence[
                "GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=9,
                message="GenerateGroundedContentResponse.Candidate.GroundingMetadata.ImageMetadata",
            )

        index: int = proto.Field(
            proto.INT32,
            number=1,
        )
        content: "GroundedGenerationContent" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="GroundedGenerationContent",
        )
        grounding_score: float = proto.Field(
            proto.FLOAT,
            number=3,
            optional=True,
        )
        grounding_metadata: "GenerateGroundedContentResponse.Candidate.GroundingMetadata" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="GenerateGroundedContentResponse.Candidate.GroundingMetadata",
        )

    candidates: MutableSequence[Candidate] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=Candidate,
    )


class CheckGroundingSpec(proto.Message):
    r"""Specification for the grounding check.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        citation_threshold (float):
            The threshold (in [0,1]) used for determining whether a fact
            must be cited for a claim in the answer candidate. Choosing
            a higher threshold will lead to fewer but very strong
            citations, while choosing a lower threshold may lead to more
            but somewhat weaker citations. If unset, the threshold will
            default to 0.6.

            This field is a member of `oneof`_ ``_citation_threshold``.
        enable_claim_level_score (bool):
            The control flag that enables claim-level
            grounding score in the response.

            This field is a member of `oneof`_ ``_enable_claim_level_score``.
    """

    citation_threshold: float = proto.Field(
        proto.DOUBLE,
        number=1,
        optional=True,
    )
    enable_claim_level_score: bool = proto.Field(
        proto.BOOL,
        number=4,
        optional=True,
    )


class CheckGroundingRequest(proto.Message):
    r"""Request message for
    [GroundedGenerationService.CheckGrounding][google.cloud.discoveryengine.v1.GroundedGenerationService.CheckGrounding]
    method.

    Attributes:
        grounding_config (str):
            Required. The resource name of the grounding config, such as
            ``projects/*/locations/global/groundingConfigs/default_grounding_config``.
        answer_candidate (str):
            Answer candidate to check. It can have a
            maximum length of 4096 tokens.
        facts (MutableSequence[google.cloud.discoveryengine_v1.types.GroundingFact]):
            List of facts for the grounding check.
            We support up to 200 facts.
        grounding_spec (google.cloud.discoveryengine_v1.types.CheckGroundingSpec):
            Configuration of the grounding check.
        user_labels (MutableMapping[str, str]):
            The user labels applied to a resource must meet the
            following requirements:

            - Each resource can have multiple labels, up to a maximum of
              64.
            - Each label must be a key-value pair.
            - Keys have a minimum length of 1 character and a maximum
              length of 63 characters and cannot be empty. Values can be
              empty and have a maximum length of 63 characters.
            - Keys and values can contain only lowercase letters,
              numeric characters, underscores, and dashes. All
              characters must use UTF-8 encoding, and international
              characters are allowed.
            - The key portion of a label must be unique. However, you
              can use the same key with multiple resources.
            - Keys must start with a lowercase letter or international
              character.

            See `Google Cloud
            Document <https://cloud.google.com/resource-manager/docs/creating-managing-labels#requirements>`__
            for more details.
    """

    grounding_config: str = proto.Field(
        proto.STRING,
        number=1,
    )
    answer_candidate: str = proto.Field(
        proto.STRING,
        number=2,
    )
    facts: MutableSequence[grounding.GroundingFact] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=grounding.GroundingFact,
    )
    grounding_spec: "CheckGroundingSpec" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="CheckGroundingSpec",
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )


class CheckGroundingResponse(proto.Message):
    r"""Response message for the
    [GroundedGenerationService.CheckGrounding][google.cloud.discoveryengine.v1.GroundedGenerationService.CheckGrounding]
    method.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        support_score (float):
            The support score for the input answer
            candidate. Higher the score, higher is the
            fraction of claims that are supported by the
            provided facts. This is always set when a
            response is returned.

            This field is a member of `oneof`_ ``_support_score``.
        cited_chunks (MutableSequence[google.cloud.discoveryengine_v1.types.FactChunk]):
            List of facts cited across all claims in the
            answer candidate. These are derived from the
            facts supplied in the request.
        cited_facts (MutableSequence[google.cloud.discoveryengine_v1.types.CheckGroundingResponse.CheckGroundingFactChunk]):
            List of facts cited across all claims in the
            answer candidate. These are derived from the
            facts supplied in the request.
        claims (MutableSequence[google.cloud.discoveryengine_v1.types.CheckGroundingResponse.Claim]):
            Claim texts and citation info across all
            claims in the answer candidate.
    """

    class CheckGroundingFactChunk(proto.Message):
        r"""Fact chunk for grounding check.

        Attributes:
            chunk_text (str):
                Text content of the fact chunk. Can be at
                most 10K characters long.
        """

        chunk_text: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class Claim(proto.Message):
        r"""Text and citation info for a claim in the answer candidate.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            start_pos (int):
                Position indicating the start of the claim in
                the answer candidate, measured in bytes. Note
                that this is not measured in characters and,
                therefore, must be rendered in the user
                interface keeping in mind that some characters
                may take more than one byte. For example, if the
                claim text contains non-ASCII characters, the
                start and end positions vary when measured in
                characters
                (programming-language-dependent) and when
                measured in bytes
                (programming-language-independent).

                This field is a member of `oneof`_ ``_start_pos``.
            end_pos (int):
                Position indicating the end of the claim in
                the answer candidate, exclusive, in bytes. Note
                that this is not measured in characters and,
                therefore, must be rendered as such. For
                example, if the claim text contains non-ASCII
                characters, the start and end positions vary
                when measured in characters
                (programming-language-dependent) and when
                measured in bytes
                (programming-language-independent).

                This field is a member of `oneof`_ ``_end_pos``.
            claim_text (str):
                Text for the claim in the answer candidate.
                Always provided regardless of whether citations
                or anti-citations are found.
            citation_indices (MutableSequence[int]):
                A list of indices (into 'cited_chunks') specifying the
                citations associated with the claim. For instance [1,3,4]
                means that cited_chunks[1], cited_chunks[3], cited_chunks[4]
                are the facts cited supporting for the claim. A citation to
                a fact indicates that the claim is supported by the fact.
            grounding_check_required (bool):
                Indicates that this claim required grounding check. When the
                system decided this claim doesn't require
                attribution/grounding check, this field will be set to
                false. In that case, no grounding check was done for the
                claim and therefore
                [citation_indices][google.cloud.discoveryengine.v1.CheckGroundingResponse.Claim.citation_indices]
                should not be returned.

                This field is a member of `oneof`_ ``_grounding_check_required``.
            score (float):
                Confidence score for the claim in the answer candidate, in
                the range of [0, 1]. This is set only when
                ``CheckGroundingRequest.grounding_spec.enable_claim_level_score``
                is true.

                This field is a member of `oneof`_ ``_score``.
        """

        start_pos: int = proto.Field(
            proto.INT32,
            number=1,
            optional=True,
        )
        end_pos: int = proto.Field(
            proto.INT32,
            number=2,
            optional=True,
        )
        claim_text: str = proto.Field(
            proto.STRING,
            number=3,
        )
        citation_indices: MutableSequence[int] = proto.RepeatedField(
            proto.INT32,
            number=4,
        )
        grounding_check_required: bool = proto.Field(
            proto.BOOL,
            number=6,
            optional=True,
        )
        score: float = proto.Field(
            proto.DOUBLE,
            number=7,
            optional=True,
        )

    support_score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    cited_chunks: MutableSequence[grounding.FactChunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=grounding.FactChunk,
    )
    cited_facts: MutableSequence[CheckGroundingFactChunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=6,
        message=CheckGroundingFactChunk,
    )
    claims: MutableSequence[Claim] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Claim,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
