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
import proto  # type: ignore

from google.cloud.monitoring_v3.types import notification

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "ListNotificationChannelDescriptorsRequest",
        "ListNotificationChannelDescriptorsResponse",
        "GetNotificationChannelDescriptorRequest",
        "CreateNotificationChannelRequest",
        "ListNotificationChannelsRequest",
        "ListNotificationChannelsResponse",
        "GetNotificationChannelRequest",
        "UpdateNotificationChannelRequest",
        "DeleteNotificationChannelRequest",
        "SendNotificationChannelVerificationCodeRequest",
        "GetNotificationChannelVerificationCodeRequest",
        "GetNotificationChannelVerificationCodeResponse",
        "VerifyNotificationChannelRequest",
    },
)


class ListNotificationChannelDescriptorsRequest(proto.Message):
    r"""The ``ListNotificationChannelDescriptors`` request.

    Attributes:
        name (str):
            Required. The REST resource name of the parent from which to
            retrieve the notification channel descriptors. The expected
            syntax is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]

            Note that this
            `names <https://cloud.google.com/monitoring/api/v3#project_name>`__
            the parent container in which to look for the descriptors;
            to retrieve a single descriptor by name, use the
            [GetNotificationChannelDescriptor][google.monitoring.v3.NotificationChannelService.GetNotificationChannelDescriptor]
            operation, instead.
        page_size (int):
            The maximum number of results to return in a
            single response. If not set to a positive
            number, a reasonable value will be chosen by the
            service.
        page_token (str):
            If non-empty, ``page_token`` must contain a value returned
            as the ``next_page_token`` in a previous response to request
            the next set of results.
    """

    name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListNotificationChannelDescriptorsResponse(proto.Message):
    r"""The ``ListNotificationChannelDescriptors`` response.

    Attributes:
        channel_descriptors (MutableSequence[google.cloud.monitoring_v3.types.NotificationChannelDescriptor]):
            The monitored resource descriptors supported
            for the specified project, optionally filtered.
        next_page_token (str):
            If not empty, indicates that there may be more results that
            match the request. Use the value in the ``page_token`` field
            in a subsequent request to fetch the next set of results. If
            empty, all results have been returned.
    """

    @property
    def raw_page(self):
        return self

    channel_descriptors: MutableSequence[
        notification.NotificationChannelDescriptor
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=notification.NotificationChannelDescriptor,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetNotificationChannelDescriptorRequest(proto.Message):
    r"""The ``GetNotificationChannelDescriptor`` response.

    Attributes:
        name (str):
            Required. The channel type for which to execute the request.
            The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/notificationChannelDescriptors/[CHANNEL_TYPE]
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateNotificationChannelRequest(proto.Message):
    r"""The ``CreateNotificationChannel`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]

            This names the container into which the channel will be
            written, this does not name the newly created channel. The
            resulting channel's name will have a normalized version of
            this field as a prefix, but will add
            ``/notificationChannels/[CHANNEL_ID]`` to identify the
            channel.
        notification_channel (google.cloud.monitoring_v3.types.NotificationChannel):
            Required. The definition of the ``NotificationChannel`` to
            create.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    notification_channel: notification.NotificationChannel = proto.Field(
        proto.MESSAGE,
        number=2,
        message=notification.NotificationChannel,
    )


class ListNotificationChannelsRequest(proto.Message):
    r"""The ``ListNotificationChannels`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]

            This names the container in which to look for the
            notification channels; it does not name a specific channel.
            To query a specific channel by REST resource name, use the
            [``GetNotificationChannel``][google.monitoring.v3.NotificationChannelService.GetNotificationChannel]
            operation.
        filter (str):
            Optional. If provided, this field specifies the criteria
            that must be met by notification channels to be included in
            the response.

            For more details, see `sorting and
            filtering <https://cloud.google.com/monitoring/api/v3/sorting-and-filtering>`__.
        order_by (str):
            Optional. A comma-separated list of fields by which to sort
            the result. Supports the same set of fields as in
            ``filter``. Entries can be prefixed with a minus sign to
            sort in descending rather than ascending order.

            For more details, see `sorting and
            filtering <https://cloud.google.com/monitoring/api/v3/sorting-and-filtering>`__.
        page_size (int):
            Optional. The maximum number of results to
            return in a single response. If not set to a
            positive number, a reasonable value will be
            chosen by the service.
        page_token (str):
            Optional. If non-empty, ``page_token`` must contain a value
            returned as the ``next_page_token`` in a previous response
            to request the next set of results.
    """

    name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=6,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=7,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListNotificationChannelsResponse(proto.Message):
    r"""The ``ListNotificationChannels`` response.

    Attributes:
        notification_channels (MutableSequence[google.cloud.monitoring_v3.types.NotificationChannel]):
            The notification channels defined for the
            specified project.
        next_page_token (str):
            If not empty, indicates that there may be more results that
            match the request. Use the value in the ``page_token`` field
            in a subsequent request to fetch the next set of results. If
            empty, all results have been returned.
        total_size (int):
            The total number of notification channels in
            all pages. This number is only an estimate, and
            may change in subsequent pages.
            https://aip.dev/158
    """

    @property
    def raw_page(self):
        return self

    notification_channels: MutableSequence[
        notification.NotificationChannel
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=notification.NotificationChannel,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    total_size: int = proto.Field(
        proto.INT32,
        number=4,
    )


class GetNotificationChannelRequest(proto.Message):
    r"""The ``GetNotificationChannel`` request.

    Attributes:
        name (str):
            Required. The channel for which to execute the request. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class UpdateNotificationChannelRequest(proto.Message):
    r"""The ``UpdateNotificationChannel`` request.

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. The fields to update.
        notification_channel (google.cloud.monitoring_v3.types.NotificationChannel):
            Required. A description of the changes to be applied to the
            specified notification channel. The description must provide
            a definition for fields to be updated; the names of these
            fields should also be included in the ``update_mask``.
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )
    notification_channel: notification.NotificationChannel = proto.Field(
        proto.MESSAGE,
        number=3,
        message=notification.NotificationChannel,
    )


class DeleteNotificationChannelRequest(proto.Message):
    r"""The ``DeleteNotificationChannel`` request.

    Attributes:
        name (str):
            Required. The channel for which to execute the request. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
        force (bool):
            If true, the notification channel will be
            deleted regardless of its use in alert policies
            (the policies will be updated to remove the
            channel). If false, this operation will fail if
            the notification channel is referenced by
            existing alerting policies.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


class SendNotificationChannelVerificationCodeRequest(proto.Message):
    r"""The ``SendNotificationChannelVerificationCode`` request.

    Attributes:
        name (str):
            Required. The notification channel to which
            to send a verification code.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetNotificationChannelVerificationCodeRequest(proto.Message):
    r"""The ``GetNotificationChannelVerificationCode`` request.

    Attributes:
        name (str):
            Required. The notification channel for which
            a verification code is to be generated and
            retrieved. This must name a channel that is
            already verified; if the specified channel is
            not verified, the request will fail.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            The desired expiration time. If specified,
            the API will guarantee that the returned code
            will not be valid after the specified timestamp;
            however, the API cannot guarantee that the
            returned code will be valid for at least as long
            as the requested time (the API puts an upper
            bound on the amount of time for which a code may
            be valid). If omitted, a default expiration will
            be used, which may be less than the max
            permissible expiration (so specifying an
            expiration may extend the code's lifetime over
            omitting an expiration, even though the API does
            impose an upper limit on the maximum expiration
            that is permitted).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class GetNotificationChannelVerificationCodeResponse(proto.Message):
    r"""The ``GetNotificationChannelVerificationCode`` request.

    Attributes:
        code (str):
            The verification code, which may be used to
            verify other channels that have an equivalent
            identity (i.e. other channels of the same type
            with the same fingerprint such as other email
            channels with the same email address or other
            sms channels with the same number).
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            The expiration time associated with the code
            that was returned. If an expiration was provided
            in the request, this is the minimum of the
            requested expiration in the request and the max
            permitted expiration.
    """

    code: str = proto.Field(
        proto.STRING,
        number=1,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class VerifyNotificationChannelRequest(proto.Message):
    r"""The ``VerifyNotificationChannel`` request.

    Attributes:
        name (str):
            Required. The notification channel to verify.
        code (str):
            Required. The verification code that was delivered to the
            channel as a result of invoking the
            ``SendNotificationChannelVerificationCode`` API method or
            that was retrieved from a verified channel via
            ``GetNotificationChannelVerificationCode``. For example, one
            might have "G-123456" or "TKNZGhhd2EyN3I1MnRnMjRv" (in
            general, one is only guaranteed that the code is valid
            UTF-8; one should not make any assumptions regarding the
            structure or format of the code).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    code: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
