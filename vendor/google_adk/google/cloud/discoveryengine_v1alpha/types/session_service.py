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

from google.cloud.discoveryengine_v1alpha.types import session

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "ListFilesRequest",
        "ListFilesResponse",
    },
)


class ListFilesRequest(proto.Message):
    r"""Request message for
    [SessionService.ListFiles][google.cloud.discoveryengine.v1alpha.SessionService.ListFiles]
    method.

    Attributes:
        parent (str):
            Required. The resource name of the Session. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/sessions/{session}``
            Name of the session resource to which the file belong.
        filter (str):
            Optional. The filter syntax consists of an expression
            language for constructing a predicate from one or more
            fields of the files being filtered. Filter expression is
            case-sensitive. Currently supported field names are:

            - upload_time
            - last_add_time
            - last_use_time
            - file_name
            - mime_type

            Some examples of filters would be:

            - "file_name = 'file_1'"
            - "file_name = 'file_1' AND mime_type = 'text/plain'"
            - "last_use_time > '2025-06-14T12:00:00Z'"

            For a full description of the filter format, please see
            https://google.aip.dev/160.
        page_size (int):
            Optional. The maximum number of files to return. The service
            may return fewer than this value. If unspecified, at most
            100 files will be returned. The maximum value is 1000;
            values above 1000 will be coerced to 1000. If user specifies
            a value less than or equal to 0 - the request will be
            rejected with an INVALID_ARGUMENT error.
        page_token (str):
            Optional. A page token received from a previous
            ``ListFiles`` call. Provide this to retrieve the subsequent
            page.

            When paginating, all other parameters provided to
            ``ListFiles`` must match the call that provided the page
            token (except ``page_size``, which may differ).
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListFilesResponse(proto.Message):
    r"""Response message for
    [SessionService.ListFiles][google.cloud.discoveryengine.v1alpha.SessionService.ListFiles]
    method.

    Attributes:
        files (MutableSequence[google.cloud.discoveryengine_v1alpha.types.FileMetadata]):
            The
            [FileMetadata][google.cloud.discoveryengine.v1alpha.FileMetadata]s.
        next_page_token (str):
            A token to retrieve next page of results. Pass this value in
            the
            [ListFilesRequest.page_token][google.cloud.discoveryengine.v1main.ListFilesRequest.page_token]
            field in the subsequent call to ``ListFiles`` method to
            retrieve the next page of results.
    """

    @property
    def raw_page(self):
        return self

    files: MutableSequence[session.FileMetadata] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=session.FileMetadata,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
