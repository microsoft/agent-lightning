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

from google.cloud.aiplatform_v1.types import cached_content as gca_cached_content
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "CreateCachedContentRequest",
        "GetCachedContentRequest",
        "UpdateCachedContentRequest",
        "DeleteCachedContentRequest",
        "ListCachedContentsRequest",
        "ListCachedContentsResponse",
    },
)


class CreateCachedContentRequest(proto.Message):
    r"""Request message for
    [GenAiCacheService.CreateCachedContent][google.cloud.aiplatform.v1.GenAiCacheService.CreateCachedContent].

    Attributes:
        parent (str):
            Required. The parent resource where the
            cached content will be created
        cached_content (google.cloud.aiplatform_v1.types.CachedContent):
            Required. The cached content to create
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    cached_content: gca_cached_content.CachedContent = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_cached_content.CachedContent,
    )


class GetCachedContentRequest(proto.Message):
    r"""Request message for
    [GenAiCacheService.GetCachedContent][google.cloud.aiplatform.v1.GenAiCacheService.GetCachedContent].

    Attributes:
        name (str):
            Required. The resource name referring to the
            cached content
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateCachedContentRequest(proto.Message):
    r"""Request message for
    [GenAiCacheService.UpdateCachedContent][google.cloud.aiplatform.v1.GenAiCacheService.UpdateCachedContent].
    Only expire_time or ttl can be updated.

    Attributes:
        cached_content (google.cloud.aiplatform_v1.types.CachedContent):
            Required. The cached content to update
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The list of fields to update.
    """

    cached_content: gca_cached_content.CachedContent = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_cached_content.CachedContent,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteCachedContentRequest(proto.Message):
    r"""Request message for
    [GenAiCacheService.DeleteCachedContent][google.cloud.aiplatform.v1.GenAiCacheService.DeleteCachedContent].

    Attributes:
        name (str):
            Required. The resource name referring to the
            cached content
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListCachedContentsRequest(proto.Message):
    r"""Request to list CachedContents.

    Attributes:
        parent (str):
            Required. The parent, which owns this
            collection of cached contents.
        page_size (int):
            Optional. The maximum number of cached
            contents to return. The service may return fewer
            than this value. If unspecified, some default
            (under maximum) number of items will be
            returned. The maximum value is 1000; values
            above 1000 will be coerced to 1000.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListCachedContents`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListCachedContents`` must match the call that provided the
            page token.
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


class ListCachedContentsResponse(proto.Message):
    r"""Response with a list of CachedContents.

    Attributes:
        cached_contents (MutableSequence[google.cloud.aiplatform_v1.types.CachedContent]):
            List of cached contents.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    cached_contents: MutableSequence[gca_cached_content.CachedContent] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gca_cached_content.CachedContent,
        )
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
