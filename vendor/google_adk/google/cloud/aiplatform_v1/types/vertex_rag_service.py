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

from google.cloud.aiplatform_v1.types import content as gca_content
from google.cloud.aiplatform_v1.types import tool
from google.cloud.aiplatform_v1.types import vertex_rag_data


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "RagQuery",
        "RetrieveContextsRequest",
        "RagContexts",
        "RetrieveContextsResponse",
        "AugmentPromptRequest",
        "AugmentPromptResponse",
        "CorroborateContentRequest",
        "CorroborateContentResponse",
        "Fact",
        "Claim",
    },
)


class RagQuery(proto.Message):
    r"""A query to retrieve relevant contexts.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        text (str):
            Optional. The query in text format to get
            relevant contexts.

            This field is a member of `oneof`_ ``query``.
        rag_retrieval_config (google.cloud.aiplatform_v1.types.RagRetrievalConfig):
            Optional. The retrieval config for the query.
    """

    text: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="query",
    )
    rag_retrieval_config: tool.RagRetrievalConfig = proto.Field(
        proto.MESSAGE,
        number=6,
        message=tool.RagRetrievalConfig,
    )


class RetrieveContextsRequest(proto.Message):
    r"""Request message for
    [VertexRagService.RetrieveContexts][google.cloud.aiplatform.v1.VertexRagService.RetrieveContexts].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        vertex_rag_store (google.cloud.aiplatform_v1.types.RetrieveContextsRequest.VertexRagStore):
            The data source for Vertex RagStore.

            This field is a member of `oneof`_ ``data_source``.
        parent (str):
            Required. The resource name of the Location from which to
            retrieve RagContexts. The users must have permission to make
            a call in the project. Format:
            ``projects/{project}/locations/{location}``.
        query (google.cloud.aiplatform_v1.types.RagQuery):
            Required. Single RAG retrieve query.
    """

    class VertexRagStore(proto.Message):
        r"""The data source for Vertex RagStore.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            rag_resources (MutableSequence[google.cloud.aiplatform_v1.types.RetrieveContextsRequest.VertexRagStore.RagResource]):
                Optional. The representation of the rag
                source. It can be used to specify corpus only or
                ragfiles. Currently only support one corpus or
                multiple files from one corpus. In the future we
                may open up multiple corpora support.
            vector_distance_threshold (float):
                Optional. Only return contexts with vector
                distance smaller than the threshold.

                This field is a member of `oneof`_ ``_vector_distance_threshold``.
        """

        class RagResource(proto.Message):
            r"""The definition of the Rag resource.

            Attributes:
                rag_corpus (str):
                    Optional. RagCorpora resource name. Format:
                    ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
                rag_file_ids (MutableSequence[str]):
                    Optional. rag_file_id. The files should be in the same
                    rag_corpus set in rag_corpus field.
            """

            rag_corpus: str = proto.Field(
                proto.STRING,
                number=1,
            )
            rag_file_ids: MutableSequence[str] = proto.RepeatedField(
                proto.STRING,
                number=2,
            )

        rag_resources: MutableSequence[
            "RetrieveContextsRequest.VertexRagStore.RagResource"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message="RetrieveContextsRequest.VertexRagStore.RagResource",
        )
        vector_distance_threshold: float = proto.Field(
            proto.DOUBLE,
            number=2,
            optional=True,
        )

    vertex_rag_store: VertexRagStore = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="data_source",
        message=VertexRagStore,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: "RagQuery" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="RagQuery",
    )


class RagContexts(proto.Message):
    r"""Relevant contexts for one query.

    Attributes:
        contexts (MutableSequence[google.cloud.aiplatform_v1.types.RagContexts.Context]):
            All its contexts.
    """

    class Context(proto.Message):
        r"""A context of the query.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            source_uri (str):
                If the file is imported from Cloud Storage or Google Drive,
                source_uri will be original file URI in Cloud Storage or
                Google Drive; if file is uploaded, source_uri will be file
                display name.
            source_display_name (str):
                The file display name.
            text (str):
                The text chunk.
            score (float):
                According to the underlying Vector DB and the selected
                metric type, the score can be either the distance or the
                similarity between the query and the context and its range
                depends on the metric type.

                For example, if the metric type is COSINE_DISTANCE, it
                represents the distance between the query and the context.
                The larger the distance, the less relevant the context is to
                the query. The range is [0, 2], while 0 means the most
                relevant and 2 means the least relevant.

                This field is a member of `oneof`_ ``_score``.
            chunk (google.cloud.aiplatform_v1.types.RagChunk):
                Context of the retrieved chunk.
        """

        source_uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        source_display_name: str = proto.Field(
            proto.STRING,
            number=5,
        )
        text: str = proto.Field(
            proto.STRING,
            number=2,
        )
        score: float = proto.Field(
            proto.DOUBLE,
            number=6,
            optional=True,
        )
        chunk: vertex_rag_data.RagChunk = proto.Field(
            proto.MESSAGE,
            number=7,
            message=vertex_rag_data.RagChunk,
        )

    contexts: MutableSequence[Context] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=Context,
    )


class RetrieveContextsResponse(proto.Message):
    r"""Response message for
    [VertexRagService.RetrieveContexts][google.cloud.aiplatform.v1.VertexRagService.RetrieveContexts].

    Attributes:
        contexts (google.cloud.aiplatform_v1.types.RagContexts):
            The contexts of the query.
    """

    contexts: "RagContexts" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RagContexts",
    )


class AugmentPromptRequest(proto.Message):
    r"""Request message for AugmentPrompt.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        vertex_rag_store (google.cloud.aiplatform_v1.types.VertexRagStore):
            Optional. Retrieves contexts from the Vertex
            RagStore.

            This field is a member of `oneof`_ ``data_source``.
        parent (str):
            Required. The resource name of the Location from which to
            augment prompt. The users must have permission to make a
            call in the project. Format:
            ``projects/{project}/locations/{location}``.
        contents (MutableSequence[google.cloud.aiplatform_v1.types.Content]):
            Optional. Input content to augment, only text
            format is supported for now.
        model (google.cloud.aiplatform_v1.types.AugmentPromptRequest.Model):
            Optional. Metadata of the backend deployed
            model.
    """

    class Model(proto.Message):
        r"""Metadata of the backend deployed model.

        Attributes:
            model (str):
                Optional. The model that the user will send
                the augmented prompt for content generation.
            model_version (str):
                Optional. The model version of the backend
                deployed model.
        """

        model: str = proto.Field(
            proto.STRING,
            number=1,
        )
        model_version: str = proto.Field(
            proto.STRING,
            number=2,
        )

    vertex_rag_store: tool.VertexRagStore = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="data_source",
        message=tool.VertexRagStore,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    contents: MutableSequence[gca_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gca_content.Content,
    )
    model: Model = proto.Field(
        proto.MESSAGE,
        number=3,
        message=Model,
    )


class AugmentPromptResponse(proto.Message):
    r"""Response message for AugmentPrompt.

    Attributes:
        augmented_prompt (MutableSequence[google.cloud.aiplatform_v1.types.Content]):
            Augmented prompt, only text format is
            supported for now.
        facts (MutableSequence[google.cloud.aiplatform_v1.types.Fact]):
            Retrieved facts from RAG data sources.
    """

    augmented_prompt: MutableSequence[gca_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_content.Content,
    )
    facts: MutableSequence["Fact"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Fact",
    )


class CorroborateContentRequest(proto.Message):
    r"""Request message for CorroborateContent.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        parent (str):
            Required. The resource name of the Location from which to
            corroborate text. The users must have permission to make a
            call in the project. Format:
            ``projects/{project}/locations/{location}``.
        content (google.cloud.aiplatform_v1.types.Content):
            Optional. Input content to corroborate, only
            text format is supported for now.

            This field is a member of `oneof`_ ``_content``.
        facts (MutableSequence[google.cloud.aiplatform_v1.types.Fact]):
            Optional. Facts used to generate the text can
            also be used to corroborate the text.
        parameters (google.cloud.aiplatform_v1.types.CorroborateContentRequest.Parameters):
            Optional. Parameters that can be set to
            override default settings per request.
    """

    class Parameters(proto.Message):
        r"""Parameters that can be overrided per request.

        Attributes:
            citation_threshold (float):
                Optional. Only return claims with citation
                score larger than the threshold.
        """

        citation_threshold: float = proto.Field(
            proto.DOUBLE,
            number=1,
        )

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    content: gca_content.Content = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message=gca_content.Content,
    )
    facts: MutableSequence["Fact"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="Fact",
    )
    parameters: Parameters = proto.Field(
        proto.MESSAGE,
        number=4,
        message=Parameters,
    )


class CorroborateContentResponse(proto.Message):
    r"""Response message for CorroborateContent.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        corroboration_score (float):
            Confidence score of corroborating content. Value is [0,1]
            with 1 is the most confidence.

            This field is a member of `oneof`_ ``_corroboration_score``.
        claims (MutableSequence[google.cloud.aiplatform_v1.types.Claim]):
            Claims that are extracted from the input
            content and facts that support the claims.
    """

    corroboration_score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    claims: MutableSequence["Claim"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Claim",
    )


class Fact(proto.Message):
    r"""The fact used in grounding.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        query (str):
            Query that is used to retrieve this fact.

            This field is a member of `oneof`_ ``_query``.
        title (str):
            If present, it refers to the title of this
            fact.

            This field is a member of `oneof`_ ``_title``.
        uri (str):
            If present, this uri links to the source of
            the fact.

            This field is a member of `oneof`_ ``_uri``.
        summary (str):
            If present, the summary/snippet of the fact.

            This field is a member of `oneof`_ ``_summary``.
        vector_distance (float):
            If present, the distance between the query
            vector and this fact vector.

            This field is a member of `oneof`_ ``_vector_distance``.
        score (float):
            If present, according to the underlying Vector DB and the
            selected metric type, the score can be either the distance
            or the similarity between the query and the fact and its
            range depends on the metric type.

            For example, if the metric type is COSINE_DISTANCE, it
            represents the distance between the query and the fact. The
            larger the distance, the less relevant the fact is to the
            query. The range is [0, 2], while 0 means the most relevant
            and 2 means the least relevant.

            This field is a member of `oneof`_ ``_score``.
        chunk (google.cloud.aiplatform_v1.types.RagChunk):
            If present, chunk properties.

            This field is a member of `oneof`_ ``_chunk``.
    """

    query: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    title: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    summary: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )
    vector_distance: float = proto.Field(
        proto.DOUBLE,
        number=5,
        optional=True,
    )
    score: float = proto.Field(
        proto.DOUBLE,
        number=6,
        optional=True,
    )
    chunk: vertex_rag_data.RagChunk = proto.Field(
        proto.MESSAGE,
        number=7,
        optional=True,
        message=vertex_rag_data.RagChunk,
    )


class Claim(proto.Message):
    r"""Claim that is extracted from the input text and facts that
    support it.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        start_index (int):
            Index in the input text where the claim
            starts (inclusive).

            This field is a member of `oneof`_ ``_start_index``.
        end_index (int):
            Index in the input text where the claim ends
            (exclusive).

            This field is a member of `oneof`_ ``_end_index``.
        fact_indexes (MutableSequence[int]):
            Indexes of the facts supporting this claim.
        score (float):
            Confidence score of this corroboration.

            This field is a member of `oneof`_ ``_score``.
    """

    start_index: int = proto.Field(
        proto.INT32,
        number=1,
        optional=True,
    )
    end_index: int = proto.Field(
        proto.INT32,
        number=2,
        optional=True,
    )
    fact_indexes: MutableSequence[int] = proto.RepeatedField(
        proto.INT32,
        number=3,
    )
    score: float = proto.Field(
        proto.FLOAT,
        number=4,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
