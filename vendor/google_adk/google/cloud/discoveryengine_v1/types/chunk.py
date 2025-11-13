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

from google.protobuf import struct_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "Chunk",
    },
)


class Chunk(proto.Message):
    r"""Chunk captures all raw metadata information of items to be
    recommended or searched in the chunk mode.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            The full resource name of the chunk. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}/documents/{document_id}/chunks/{chunk_id}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        id (str):
            Unique chunk ID of the current chunk.
        content (str):
            Content is a string from a document (parsed
            content).
        relevance_score (float):
            Output only. Represents the relevance score based on
            similarity. Higher score indicates higher chunk relevance.
            The score is in range [-1.0, 1.0]. Only populated on
            [SearchResponse][google.cloud.discoveryengine.v1.SearchResponse].

            This field is a member of `oneof`_ ``_relevance_score``.
        document_metadata (google.cloud.discoveryengine_v1.types.Chunk.DocumentMetadata):
            Metadata of the document from the current
            chunk.
        derived_struct_data (google.protobuf.struct_pb2.Struct):
            Output only. This field is OUTPUT_ONLY. It contains derived
            data that are not in the original input document.
        page_span (google.cloud.discoveryengine_v1.types.Chunk.PageSpan):
            Page span of the chunk.
        chunk_metadata (google.cloud.discoveryengine_v1.types.Chunk.ChunkMetadata):
            Output only. Metadata of the current chunk.
        data_urls (MutableSequence[str]):
            Output only. Image Data URLs if the current chunk contains
            images. Data URLs are composed of four parts: a prefix
            (data:), a MIME type indicating the type of data, an
            optional base64 token if non-textual, and the data itself:
            data:[][;base64],
        annotation_contents (MutableSequence[str]):
            Output only. Annotation contents if the
            current chunk contains annotations.
        annotation_metadata (MutableSequence[google.cloud.discoveryengine_v1.types.Chunk.AnnotationMetadata]):
            Output only. The annotation metadata includes
            structured content in the current chunk.
    """

    class StructureType(proto.Enum):
        r"""Defines the types of the structured content that can be
        extracted.

        Values:
            STRUCTURE_TYPE_UNSPECIFIED (0):
                Default value.
            SHAREHOLDER_STRUCTURE (1):
                Shareholder structure.
            SIGNATURE_STRUCTURE (2):
                Signature structure.
            CHECKBOX_STRUCTURE (3):
                Checkbox structure.
        """
        STRUCTURE_TYPE_UNSPECIFIED = 0
        SHAREHOLDER_STRUCTURE = 1
        SIGNATURE_STRUCTURE = 2
        CHECKBOX_STRUCTURE = 3

    class DocumentMetadata(proto.Message):
        r"""Document metadata contains the information of the document of
        the current chunk.

        Attributes:
            uri (str):
                Uri of the document.
            title (str):
                Title of the document.
            struct_data (google.protobuf.struct_pb2.Struct):
                Data representation. The structured JSON data for the
                document. It should conform to the registered
                [Schema][google.cloud.discoveryengine.v1.Schema] or an
                ``INVALID_ARGUMENT`` error is thrown.
        """

        uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        title: str = proto.Field(
            proto.STRING,
            number=2,
        )
        struct_data: struct_pb2.Struct = proto.Field(
            proto.MESSAGE,
            number=3,
            message=struct_pb2.Struct,
        )

    class PageSpan(proto.Message):
        r"""Page span of the chunk.

        Attributes:
            page_start (int):
                The start page of the chunk.
            page_end (int):
                The end page of the chunk.
        """

        page_start: int = proto.Field(
            proto.INT32,
            number=1,
        )
        page_end: int = proto.Field(
            proto.INT32,
            number=2,
        )

    class ChunkMetadata(proto.Message):
        r"""Metadata of the current chunk. This field is only populated on
        [SearchService.Search][google.cloud.discoveryengine.v1.SearchService.Search]
        API.

        Attributes:
            previous_chunks (MutableSequence[google.cloud.discoveryengine_v1.types.Chunk]):
                The previous chunks of the current chunk. The number is
                controlled by
                [SearchRequest.ContentSearchSpec.ChunkSpec.num_previous_chunks][google.cloud.discoveryengine.v1.SearchRequest.ContentSearchSpec.ChunkSpec.num_previous_chunks].
                This field is only populated on
                [SearchService.Search][google.cloud.discoveryengine.v1.SearchService.Search]
                API.
            next_chunks (MutableSequence[google.cloud.discoveryengine_v1.types.Chunk]):
                The next chunks of the current chunk. The number is
                controlled by
                [SearchRequest.ContentSearchSpec.ChunkSpec.num_next_chunks][google.cloud.discoveryengine.v1.SearchRequest.ContentSearchSpec.ChunkSpec.num_next_chunks].
                This field is only populated on
                [SearchService.Search][google.cloud.discoveryengine.v1.SearchService.Search]
                API.
        """

        previous_chunks: MutableSequence["Chunk"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="Chunk",
        )
        next_chunks: MutableSequence["Chunk"] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="Chunk",
        )

    class StructuredContent(proto.Message):
        r"""The structured content information.

        Attributes:
            structure_type (google.cloud.discoveryengine_v1.types.Chunk.StructureType):
                Output only. The structure type of the
                structured content.
            content (str):
                Output only. The content of the structured
                content.
        """

        structure_type: "Chunk.StructureType" = proto.Field(
            proto.ENUM,
            number=1,
            enum="Chunk.StructureType",
        )
        content: str = proto.Field(
            proto.STRING,
            number=2,
        )

    class AnnotationMetadata(proto.Message):
        r"""The annotation metadata includes structured content in the
        current chunk.

        Attributes:
            structured_content (google.cloud.discoveryengine_v1.types.Chunk.StructuredContent):
                Output only. The structured content
                information.
            image_id (str):
                Output only. Image id is provided if the
                structured content is based on an image.
        """

        structured_content: "Chunk.StructuredContent" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Chunk.StructuredContent",
        )
        image_id: str = proto.Field(
            proto.STRING,
            number=2,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    content: str = proto.Field(
        proto.STRING,
        number=3,
    )
    relevance_score: float = proto.Field(
        proto.DOUBLE,
        number=8,
        optional=True,
    )
    document_metadata: DocumentMetadata = proto.Field(
        proto.MESSAGE,
        number=5,
        message=DocumentMetadata,
    )
    derived_struct_data: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=4,
        message=struct_pb2.Struct,
    )
    page_span: PageSpan = proto.Field(
        proto.MESSAGE,
        number=6,
        message=PageSpan,
    )
    chunk_metadata: ChunkMetadata = proto.Field(
        proto.MESSAGE,
        number=7,
        message=ChunkMetadata,
    )
    data_urls: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=9,
    )
    annotation_contents: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )
    annotation_metadata: MutableSequence[AnnotationMetadata] = proto.RepeatedField(
        proto.MESSAGE,
        number=12,
        message=AnnotationMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
