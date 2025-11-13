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
    package="google.cloud.discoveryengine.v1beta",
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
            [SearchService.SearchResponse][].

            This field is a member of `oneof`_ ``_relevance_score``.
        document_metadata (google.cloud.discoveryengine_v1beta.types.Chunk.DocumentMetadata):
            Metadata of the document from the current
            chunk.
        derived_struct_data (google.protobuf.struct_pb2.Struct):
            Output only. This field is OUTPUT_ONLY. It contains derived
            data that are not in the original input document.
        page_span (google.cloud.discoveryengine_v1beta.types.Chunk.PageSpan):
            Page span of the chunk.
        chunk_metadata (google.cloud.discoveryengine_v1beta.types.Chunk.ChunkMetadata):
            Output only. Metadata of the current chunk.
    """

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
                [Schema][google.cloud.discoveryengine.v1beta.Schema] or an
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
        [SearchService.Search][google.cloud.discoveryengine.v1beta.SearchService.Search]
        API.

        Attributes:
            previous_chunks (MutableSequence[google.cloud.discoveryengine_v1beta.types.Chunk]):
                The previous chunks of the current chunk. The number is
                controlled by
                [SearchRequest.ContentSearchSpec.ChunkSpec.num_previous_chunks][google.cloud.discoveryengine.v1beta.SearchRequest.ContentSearchSpec.ChunkSpec.num_previous_chunks].
                This field is only populated on
                [SearchService.Search][google.cloud.discoveryengine.v1beta.SearchService.Search]
                API.
            next_chunks (MutableSequence[google.cloud.discoveryengine_v1beta.types.Chunk]):
                The next chunks of the current chunk. The number is
                controlled by
                [SearchRequest.ContentSearchSpec.ChunkSpec.num_next_chunks][google.cloud.discoveryengine.v1beta.SearchRequest.ContentSearchSpec.ChunkSpec.num_next_chunks].
                This field is only populated on
                [SearchService.Search][google.cloud.discoveryengine.v1beta.SearchService.Search]
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


__all__ = tuple(sorted(__protobuf__.manifest))
