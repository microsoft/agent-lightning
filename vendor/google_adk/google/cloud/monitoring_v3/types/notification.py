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

from google.api import label_pb2  # type: ignore
from google.api import launch_stage_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.monitoring_v3.types import common, mutation_record

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "NotificationChannelDescriptor",
        "NotificationChannel",
    },
)


class NotificationChannelDescriptor(proto.Message):
    r"""A description of a notification channel. The descriptor
    includes the properties of the channel and the set of labels or
    fields that must be specified to configure channels of a given
    type.

    Attributes:
        name (str):
            The full REST resource name for this descriptor. The format
            is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/notificationChannelDescriptors/[TYPE]

            In the above, ``[TYPE]`` is the value of the ``type`` field.
        type_ (str):
            The type of notification channel, such as "email" and "sms".
            To view the full list of channels, see `Channel
            descriptors <https://cloud.google.com/monitoring/alerts/using-channels-api#ncd>`__.
            Notification channel types are globally unique.
        display_name (str):
            A human-readable name for the notification
            channel type.  This form of the name is suitable
            for a user interface.
        description (str):
            A human-readable description of the
            notification channel type. The description may
            include a description of the properties of the
            channel and pointers to external documentation.
        labels (MutableSequence[google.api.label_pb2.LabelDescriptor]):
            The set of labels that must be defined to
            identify a particular channel of the
            corresponding type. Each label includes a
            description for how that field should be
            populated.
        supported_tiers (MutableSequence[google.cloud.monitoring_v3.types.ServiceTier]):
            The tiers that support this notification channel; the
            project service tier must be one of the supported_tiers.
        launch_stage (google.api.launch_stage_pb2.LaunchStage):
            The product launch stage for channels of this
            type.
    """

    name: str = proto.Field(
        proto.STRING,
        number=6,
    )
    type_: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    labels: MutableSequence[label_pb2.LabelDescriptor] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=label_pb2.LabelDescriptor,
    )
    supported_tiers: MutableSequence[common.ServiceTier] = proto.RepeatedField(
        proto.ENUM,
        number=5,
        enum=common.ServiceTier,
    )
    launch_stage: launch_stage_pb2.LaunchStage = proto.Field(
        proto.ENUM,
        number=7,
        enum=launch_stage_pb2.LaunchStage,
    )


class NotificationChannel(proto.Message):
    r"""A ``NotificationChannel`` is a medium through which an alert is
    delivered when a policy violation is detected. Examples of channels
    include email, SMS, and third-party messaging applications. Fields
    containing sensitive information like authentication tokens or
    contact info are only partially populated on retrieval.

    Attributes:
        type_ (str):
            The type of the notification channel. This field matches the
            value of the
            [NotificationChannelDescriptor.type][google.monitoring.v3.NotificationChannelDescriptor.type]
            field.
        name (str):
            Identifier. The full REST resource name for this channel.
            The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]

            The ``[CHANNEL_ID]`` is automatically assigned by the server
            on creation.
        display_name (str):
            An optional human-readable name for this
            notification channel. It is recommended that you
            specify a non-empty and unique name in order to
            make it easier to identify the channels in your
            project, though this is not enforced. The
            display name is limited to 512 Unicode
            characters.
        description (str):
            An optional human-readable description of
            this notification channel. This description may
            provide additional details, beyond the display
            name, for the channel. This may not exceed 1024
            Unicode characters.
        labels (MutableMapping[str, str]):
            Configuration fields that define the channel and its
            behavior. The permissible and required labels are specified
            in the
            [NotificationChannelDescriptor.labels][google.monitoring.v3.NotificationChannelDescriptor.labels]
            of the ``NotificationChannelDescriptor`` corresponding to
            the ``type`` field.
        user_labels (MutableMapping[str, str]):
            User-supplied key/value data that does not need to conform
            to the corresponding ``NotificationChannelDescriptor``'s
            schema, unlike the ``labels`` field. This field is intended
            to be used for organizing and identifying the
            ``NotificationChannel`` objects.

            The field can contain up to 64 entries. Each key and value
            is limited to 63 Unicode characters or 128 bytes, whichever
            is smaller. Labels and values can contain only lowercase
            letters, numerals, underscores, and dashes. Keys must begin
            with a letter.
        verification_status (google.cloud.monitoring_v3.types.NotificationChannel.VerificationStatus):
            Indicates whether this channel has been verified or not. On
            a
            [``ListNotificationChannels``][google.monitoring.v3.NotificationChannelService.ListNotificationChannels]
            or
            [``GetNotificationChannel``][google.monitoring.v3.NotificationChannelService.GetNotificationChannel]
            operation, this field is expected to be populated.

            If the value is ``UNVERIFIED``, then it indicates that the
            channel is non-functioning (it both requires verification
            and lacks verification); otherwise, it is assumed that the
            channel works.

            If the channel is neither ``VERIFIED`` nor ``UNVERIFIED``,
            it implies that the channel is of a type that does not
            require verification or that this specific channel has been
            exempted from verification because it was created prior to
            verification being required for channels of this type.

            This field cannot be modified using a standard
            [``UpdateNotificationChannel``][google.monitoring.v3.NotificationChannelService.UpdateNotificationChannel]
            operation. To change the value of this field, you must call
            [``VerifyNotificationChannel``][google.monitoring.v3.NotificationChannelService.VerifyNotificationChannel].
        enabled (google.protobuf.wrappers_pb2.BoolValue):
            Whether notifications are forwarded to the
            described channel. This makes it possible to
            disable delivery of notifications to a
            particular channel without removing the channel
            from all alerting policies that reference the
            channel. This is a more convenient approach when
            the change is temporary and you want to receive
            notifications from the same set of alerting
            policies on the channel at some point in the
            future.
        creation_record (google.cloud.monitoring_v3.types.MutationRecord):
            Record of the creation of this channel.
        mutation_records (MutableSequence[google.cloud.monitoring_v3.types.MutationRecord]):
            Records of the modification of this channel.
    """

    class VerificationStatus(proto.Enum):
        r"""Indicates whether the channel has been verified or not. It is
        illegal to specify this field in a
        [``CreateNotificationChannel``][google.monitoring.v3.NotificationChannelService.CreateNotificationChannel]
        or an
        [``UpdateNotificationChannel``][google.monitoring.v3.NotificationChannelService.UpdateNotificationChannel]
        operation.

        Values:
            VERIFICATION_STATUS_UNSPECIFIED (0):
                Sentinel value used to indicate that the
                state is unknown, omitted, or is not applicable
                (as in the case of channels that neither support
                nor require verification in order to function).
            UNVERIFIED (1):
                The channel has yet to be verified and
                requires verification to function. Note that
                this state also applies to the case where the
                verification process has been initiated by
                sending a verification code but where the
                verification code has not been submitted to
                complete the process.
            VERIFIED (2):
                It has been proven that notifications can be
                received on this notification channel and that
                someone on the project has access to messages
                that are delivered to that channel.
        """
        VERIFICATION_STATUS_UNSPECIFIED = 0
        UNVERIFIED = 1
        VERIFIED = 2

    type_: str = proto.Field(
        proto.STRING,
        number=1,
    )
    name: str = proto.Field(
        proto.STRING,
        number=6,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    description: str = proto.Field(
        proto.STRING,
        number=4,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=8,
    )
    verification_status: VerificationStatus = proto.Field(
        proto.ENUM,
        number=9,
        enum=VerificationStatus,
    )
    enabled: wrappers_pb2.BoolValue = proto.Field(
        proto.MESSAGE,
        number=11,
        message=wrappers_pb2.BoolValue,
    )
    creation_record: mutation_record.MutationRecord = proto.Field(
        proto.MESSAGE,
        number=12,
        message=mutation_record.MutationRecord,
    )
    mutation_records: MutableSequence[
        mutation_record.MutationRecord
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=13,
        message=mutation_record.MutationRecord,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
