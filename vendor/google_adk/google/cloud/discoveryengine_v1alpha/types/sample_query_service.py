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

from google.protobuf import field_mask_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1alpha.types import sample_query as gcd_sample_query

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GetSampleQueryRequest",
        "ListSampleQueriesRequest",
        "ListSampleQueriesResponse",
        "CreateSampleQueryRequest",
        "UpdateSampleQueryRequest",
        "DeleteSampleQueryRequest",
    },
)


class GetSampleQueryRequest(proto.Message):
    r"""Request message for
    [SampleQueryService.GetSampleQuery][google.cloud.discoveryengine.v1alpha.SampleQueryService.GetSampleQuery]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            such as
            ``projects/{project}/locations/{location}/sampleQuerySets/{sample_query_set}/sampleQueries/{sample_query}``.

            If the caller does not have permission to access the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the requested
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]
            does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListSampleQueriesRequest(proto.Message):
    r"""Request message for
    [SampleQueryService.ListSampleQueries][google.cloud.discoveryengine.v1alpha.SampleQueryService.ListSampleQueries]
    method.

    Attributes:
        parent (str):
            Required. The parent sample query set resource name, such as
            ``projects/{project}/locations/{location}/sampleQuerySets/{sampleQuerySet}``.

            If the caller does not have permission to list
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]s
            under this sample query set, regardless of whether or not
            this sample query set exists, a ``PERMISSION_DENIED`` error
            is returned.
        page_size (int):
            Maximum number of
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]s
            to return. If unspecified, defaults to 100. The maximum
            allowed value is 1000. Values above 1000 will be coerced to
            1000.

            If this field is negative, an ``INVALID_ARGUMENT`` error is
            returned.
        page_token (str):
            A page token
            [ListSampleQueriesResponse.next_page_token][google.cloud.discoveryengine.v1alpha.ListSampleQueriesResponse.next_page_token],
            received from a previous
            [SampleQueryService.ListSampleQueries][google.cloud.discoveryengine.v1alpha.SampleQueryService.ListSampleQueries]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [SampleQueryService.ListSampleQueries][google.cloud.discoveryengine.v1alpha.SampleQueryService.ListSampleQueries]
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


class ListSampleQueriesResponse(proto.Message):
    r"""Response message for
    [SampleQueryService.ListSampleQueries][google.cloud.discoveryengine.v1alpha.SampleQueryService.ListSampleQueries]
    method.

    Attributes:
        sample_queries (MutableSequence[google.cloud.discoveryengine_v1alpha.types.SampleQuery]):
            The
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]s.
        next_page_token (str):
            A token that can be sent as
            [ListSampleQueriesRequest.page_token][google.cloud.discoveryengine.v1alpha.ListSampleQueriesRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    sample_queries: MutableSequence[gcd_sample_query.SampleQuery] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_sample_query.SampleQuery,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateSampleQueryRequest(proto.Message):
    r"""Request message for
    [SampleQueryService.CreateSampleQuery][google.cloud.discoveryengine.v1alpha.SampleQueryService.CreateSampleQuery]
    method.

    Attributes:
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}/sampleQuerySets/{sampleQuerySet}``.
        sample_query (google.cloud.discoveryengine_v1alpha.types.SampleQuery):
            Required. The
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]
            to create.
        sample_query_id (str):
            Required. The ID to use for the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            which will become the final component of the
            [SampleQuery.name][google.cloud.discoveryengine.v1alpha.SampleQuery.name].

            If the caller does not have permission to create the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            This field must be unique among all
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]s
            with the same
            [parent][google.cloud.discoveryengine.v1alpha.CreateSampleQueryRequest.parent].
            Otherwise, an ``ALREADY_EXISTS`` error is returned.

            This field must conform to
            `RFC-1034 <https://tools.ietf.org/html/rfc1034>`__ standard
            with a length limit of 63 characters. Otherwise, an
            ``INVALID_ARGUMENT`` error is returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    sample_query: gcd_sample_query.SampleQuery = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_sample_query.SampleQuery,
    )
    sample_query_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class UpdateSampleQueryRequest(proto.Message):
    r"""Request message for
    [SampleQueryService.UpdateSampleQuery][google.cloud.discoveryengine.v1alpha.SampleQueryService.UpdateSampleQuery]
    method.

    Attributes:
        sample_query (google.cloud.discoveryengine_v1alpha.types.SampleQuery):
            Required. The simple query to update.

            If the caller does not have permission to update the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]
            to update does not exist a ``NOT_FOUND`` error is returned.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            imported 'simple query' to update. If not set,
            will by default update all fields.
    """

    sample_query: gcd_sample_query.SampleQuery = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_sample_query.SampleQuery,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteSampleQueryRequest(proto.Message):
    r"""Request message for
    [SampleQueryService.DeleteSampleQuery][google.cloud.discoveryengine.v1alpha.SampleQueryService.DeleteSampleQuery]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            such as
            ``projects/{project}/locations/{location}/sampleQuerySets/{sample_query_set}/sampleQueries/{sample_query}``.

            If the caller does not have permission to delete the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery],
            regardless of whether or not it exists, a
            ``PERMISSION_DENIED`` error is returned.

            If the
            [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]
            to delete does not exist, a ``NOT_FOUND`` error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
