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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ReservationAffinity",
    },
)


class ReservationAffinity(proto.Message):
    r"""A ReservationAffinity can be used to configure a Vertex AI
    resource (e.g., a DeployedModel) to draw its Compute Engine
    resources from a Shared Reservation, or exclusively from
    on-demand capacity.

    Attributes:
        reservation_affinity_type (google.cloud.aiplatform_v1beta1.types.ReservationAffinity.Type):
            Required. Specifies the reservation affinity
            type.
        key (str):
            Optional. Corresponds to the label key of a reservation
            resource. To target a SPECIFIC_RESERVATION by name, use
            ``compute.googleapis.com/reservation-name`` as the key and
            specify the name of your reservation as its value.
        values (MutableSequence[str]):
            Optional. Corresponds to the label values of
            a reservation resource. This must be the full
            resource name of the reservation.
    """

    class Type(proto.Enum):
        r"""Identifies a type of reservation affinity.

        Values:
            TYPE_UNSPECIFIED (0):
                Default value. This should not be used.
            NO_RESERVATION (1):
                Do not consume from any reserved capacity,
                only use on-demand.
            ANY_RESERVATION (2):
                Consume any reservation available, falling
                back to on-demand.
            SPECIFIC_RESERVATION (3):
                Consume from a specific reservation. When chosen, the
                reservation must be identified via the ``key`` and
                ``values`` fields.
        """

        TYPE_UNSPECIFIED = 0
        NO_RESERVATION = 1
        ANY_RESERVATION = 2
        SPECIFIC_RESERVATION = 3

    reservation_affinity_type: Type = proto.Field(
        proto.ENUM,
        number=1,
        enum=Type,
    )
    key: str = proto.Field(
        proto.STRING,
        number=2,
    )
    values: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
