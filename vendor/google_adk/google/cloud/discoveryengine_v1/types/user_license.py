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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "UserLicense",
    },
)


class UserLicense(proto.Message):
    r"""User License information assigned by the admin.

    Attributes:
        user_principal (str):
            Required. Immutable. The user principal of
            the User, could be email address or other
            prinical identifier. This field is immutable.
            Admin assign licenses based on the user
            principal.
        user_profile (str):
            Optional. The user profile.
            We user user full name(First name + Last name)
            as user profile.
        license_assignment_state (google.cloud.discoveryengine_v1.types.UserLicense.LicenseAssignmentState):
            Output only. License assignment state of the
            user. If the user is assigned with a license
            config, the user loggin will be assigned with
            the license;
            If the user's license assignment state is
            unassigned or unspecified, no license config
            will be associated to the user;
        license_config (str):
            Optional. The full resource name of the
            Subscription(LicenseConfig) assigned to the
            user.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. User created timestamp.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. User update timestamp.
        last_login_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. User last logged in time.
            If the user has not logged in yet, this field
            will be empty.
    """

    class LicenseAssignmentState(proto.Enum):
        r"""License assignment state enumeration.

        Values:
            LICENSE_ASSIGNMENT_STATE_UNSPECIFIED (0):
                Default value.
            ASSIGNED (1):
                License assigned to the user.
            UNASSIGNED (2):
                No license assigned to the user. Deprecated, translated to
                NO_LICENSE.
            NO_LICENSE (3):
                No license assigned to the user.
            NO_LICENSE_ATTEMPTED_LOGIN (4):
                User attempted to login but no license assigned to the user.
                This state is only used for no user first time login attempt
                but cannot get license assigned. Users already logged in but
                cannot get license assigned will be assigned NO_LICENSE
                state(License could be unassigned by admin).
        """
        LICENSE_ASSIGNMENT_STATE_UNSPECIFIED = 0
        ASSIGNED = 1
        UNASSIGNED = 2
        NO_LICENSE = 3
        NO_LICENSE_ATTEMPTED_LOGIN = 4

    user_principal: str = proto.Field(
        proto.STRING,
        number=1,
    )
    user_profile: str = proto.Field(
        proto.STRING,
        number=3,
    )
    license_assignment_state: LicenseAssignmentState = proto.Field(
        proto.ENUM,
        number=4,
        enum=LicenseAssignmentState,
    )
    license_config: str = proto.Field(
        proto.STRING,
        number=5,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    last_login_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
