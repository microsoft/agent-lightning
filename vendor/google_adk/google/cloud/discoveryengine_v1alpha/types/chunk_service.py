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

from google.cloud.discoveryengine_v1alpha.types import chunk

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GetChunkRequest",
        "ListChunksRequest",
        "ListChunksResponse",
    },
)


class GetChunkRequest(proto.Message):
    r"""Request message for
    [ChunkService.GetChunk][google.cloud.discoveryengine.v1alpha.ChunkService.GetChunk]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Chunk][google.cloud.discoveryengine.v1alpha.Chunk], such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}/documents/{document}/chunks/{chunk}``.

            If the caller does not have permission to access the
            [Chunk][google.cloud.discoveryengine.v1alpha.Chunk],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the requested
            [Chunk][google.cloud.discoveryengine.v1alpha.Chunk] does not
            exist, a ``NOT_FOUND`` error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListChunksRequest(proto.Message):
    r"""Request message for
    [ChunkService.ListChunks][google.cloud.discoveryengine.v1alpha.ChunkService.ListChunks]
    method.

    Attributes:
        parent (str):
            Required. The parent document resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}/documents/{document}``.

            If the caller does not have permission to list
            [Chunk][google.cloud.discoveryengine.v1alpha.Chunk]s under
            this document, regardless of whether or not this document
            exists, a ``PERMISSION_DENIED`` error is returned.
        page_size (int):
            Maximum number of
            [Chunk][google.cloud.discoveryengine.v1alpha.Chunk]s to
            return. If unspecified, defaults to 100. The maximum allowed
            value is 1000. Values above 1000 will be coerced to 1000.

            If this field is negative, an ``INVALID_ARGUMENT`` error is
            returned.
        page_token (str):
            A page token
            [ListChunksResponse.next_page_token][google.cloud.discoveryengine.v1alpha.ListChunksResponse.next_page_token],
            received from a previous
            [ChunkService.ListChunks][google.cloud.discoveryengine.v1alpha.ChunkService.ListChunks]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [ChunkService.ListChunks][google.cloud.discoveryengine.v1alpha.ChunkService.ListChunks]
            must match the call that provided the page token. Otherwise,
            an ``INVALID_ARGUMENT`` error is returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListChunksResponse(proto.Message):
    r"""Response message for
    [ChunkService.ListChunks][google.cloud.discoveryengine.v1alpha.ChunkService.ListChunks]
    method.

    Attributes:
        chunks (MutableSequence[google.cloud.discoveryengine_v1alpha.types.Chunk]):
            The [Chunk][google.cloud.discoveryengine.v1alpha.Chunk]s.
        next_page_token (str):
            A token that can be sent as
            [ListChunksRequest.page_token][google.cloud.discoveryengine.v1alpha.ListChunksRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    chunks: MutableSequence[chunk.Chunk] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=chunk.Chunk,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
