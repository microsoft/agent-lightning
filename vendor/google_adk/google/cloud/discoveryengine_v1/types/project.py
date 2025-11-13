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
        "Project",
    },
)


class Project(proto.Message):
    r"""Metadata and configurations for a Google Cloud project in the
    service.

    Attributes:
        name (str):
            Output only. Full resource name of the project, for example
            ``projects/{project}``. Note that when making requests,
            project number and project id are both acceptable, but the
            server will always respond in project number.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when this project
            is created.
        provision_completion_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when this project
            is successfully provisioned. Empty value means
            this project is still provisioning and is not
            ready for use.
        service_terms_map (MutableMapping[str, google.cloud.discoveryengine_v1.types.Project.ServiceTerms]):
            Output only. A map of terms of services. The key is the
            ``id`` of
            [ServiceTerms][google.cloud.discoveryengine.v1.Project.ServiceTerms].
    """

    class ServiceTerms(proto.Message):
        r"""Metadata about the terms of service.

        Attributes:
            id (str):
                The unique identifier of this terms of service. Available
                terms:

                - ``GA_DATA_USE_TERMS``: `Terms for data
                  use <https://cloud.google.com/retail/data-use-terms>`__.
                  When using this as ``id``, the acceptable
                  [version][google.cloud.discoveryengine.v1.Project.ServiceTerms.version]
                  to provide is ``2022-11-23``.
            version (str):
                The version string of the terms of service. For acceptable
                values, see the comments for
                [id][google.cloud.discoveryengine.v1.Project.ServiceTerms.id]
                above.
            state (google.cloud.discoveryengine_v1.types.Project.ServiceTerms.State):
                Whether the project has accepted/rejected the
                service terms or it is still pending.
            accept_time (google.protobuf.timestamp_pb2.Timestamp):
                The last time when the project agreed to the
                terms of service.
            decline_time (google.protobuf.timestamp_pb2.Timestamp):
                The last time when the project declined or
                revoked the agreement to terms of service.
        """

        class State(proto.Enum):
            r"""The agreement states this terms of service.

            Values:
                STATE_UNSPECIFIED (0):
                    The default value of the enum. This value is
                    not actually used.
                TERMS_ACCEPTED (1):
                    The project has given consent to the terms of
                    service.
                TERMS_PENDING (2):
                    The project is pending to review and accept
                    the terms of service.
                TERMS_DECLINED (3):
                    The project has declined or revoked the
                    agreement to terms of service.
            """
            STATE_UNSPECIFIED = 0
            TERMS_ACCEPTED = 1
            TERMS_PENDING = 2
            TERMS_DECLINED = 3

        id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        version: str = proto.Field(
            proto.STRING,
            number=2,
        )
        state: "Project.ServiceTerms.State" = proto.Field(
            proto.ENUM,
            number=4,
            enum="Project.ServiceTerms.State",
        )
        accept_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=5,
            message=timestamp_pb2.Timestamp,
        )
        decline_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=6,
            message=timestamp_pb2.Timestamp,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    provision_completion_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    service_terms_map: MutableMapping[str, ServiceTerms] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=4,
        message=ServiceTerms,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
