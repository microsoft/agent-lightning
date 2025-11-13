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
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import user_license

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "ListUserLicensesRequest",
        "ListUserLicensesResponse",
        "BatchUpdateUserLicensesRequest",
        "BatchUpdateUserLicensesMetadata",
        "BatchUpdateUserLicensesResponse",
    },
)


class ListUserLicensesRequest(proto.Message):
    r"""Request message for
    [UserLicenseService.ListUserLicenses][google.cloud.discoveryengine.v1.UserLicenseService.ListUserLicenses].

    Attributes:
        parent (str):
            Required. The parent [UserStore][] resource name, format:
            ``projects/{project}/locations/{location}/userStores/{user_store_id}``.
        page_size (int):
            Optional. Requested page size. Server may return fewer items
            than requested. If unspecified, defaults to 10. The maximum
            value is 50; values above 50 will be coerced to 50.

            If this field is negative, an INVALID_ARGUMENT error is
            returned.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListUserLicenses`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListUserLicenses`` must match the call that provided the
            page token.
        filter (str):
            Optional. Filter for the list request.

            Supported fields:

            - ``license_assignment_state``

            Examples:

            - ``license_assignment_state = ASSIGNED`` to list assigned
              user licenses.
            - ``license_assignment_state = NO_LICENSE`` to list not
              licensed users.
            - ``license_assignment_state = NO_LICENSE_ATTEMPTED_LOGIN``
              to list users who attempted login but no license assigned.
            - ``license_assignment_state != NO_LICENSE_ATTEMPTED_LOGIN``
              to filter out users who attempted login but no license
              assigned.
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
    filter: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListUserLicensesResponse(proto.Message):
    r"""Response message for
    [UserLicenseService.ListUserLicenses][google.cloud.discoveryengine.v1.UserLicenseService.ListUserLicenses].

    Attributes:
        user_licenses (MutableSequence[google.cloud.discoveryengine_v1.types.UserLicense]):
            All the customer's
            [UserLicense][google.cloud.discoveryengine.v1.UserLicense]s.
        next_page_token (str):
            A token, which can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    user_licenses: MutableSequence[user_license.UserLicense] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=user_license.UserLicense,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class BatchUpdateUserLicensesRequest(proto.Message):
    r"""Request message for
    [UserLicenseService.BatchUpdateUserLicenses][google.cloud.discoveryengine.v1.UserLicenseService.BatchUpdateUserLicenses]
    method.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.BatchUpdateUserLicensesRequest.InlineSource):
            The inline source for the input content for
            document embeddings.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. The parent [UserStore][] resource name, format:
            ``projects/{project}/locations/{location}/userStores/{user_store_id}``.
        delete_unassigned_user_licenses (bool):
            Optional. If true, if user licenses removed
            associated license config, the user license will
            be deleted. By default which is false, the user
            license will be updated to unassigned state.
    """

    class InlineSource(proto.Message):
        r"""The inline source for the input config for
        BatchUpdateUserLicenses method.

        Attributes:
            user_licenses (MutableSequence[google.cloud.discoveryengine_v1.types.UserLicense]):
                Required. A list of user licenses to update. Each user
                license must have a valid
                [UserLicense.user_principal][google.cloud.discoveryengine.v1.UserLicense.user_principal].
            update_mask (google.protobuf.field_mask_pb2.FieldMask):
                Optional. The list of fields to update.
        """

        user_licenses: MutableSequence[user_license.UserLicense] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=user_license.UserLicense,
        )
        update_mask: field_mask_pb2.FieldMask = proto.Field(
            proto.MESSAGE,
            number=2,
            message=field_mask_pb2.FieldMask,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    delete_unassigned_user_licenses: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class BatchUpdateUserLicensesMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [UserLicenseService.BatchUpdateUserLicenses][google.cloud.discoveryengine.v1.UserLicenseService.BatchUpdateUserLicenses]
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        success_count (int):
            Count of user licenses successfully updated.
        failure_count (int):
            Count of user licenses that failed to be
            updated.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    success_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=4,
    )


class BatchUpdateUserLicensesResponse(proto.Message):
    r"""Response message for
    [UserLicenseService.BatchUpdateUserLicenses][google.cloud.discoveryengine.v1.UserLicenseService.BatchUpdateUserLicenses]
    method.

    Attributes:
        user_licenses (MutableSequence[google.cloud.discoveryengine_v1.types.UserLicense]):
            UserLicenses successfully updated.
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
    """

    user_licenses: MutableSequence[user_license.UserLicense] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=user_license.UserLicense,
    )
    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=status_pb2.Status,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
