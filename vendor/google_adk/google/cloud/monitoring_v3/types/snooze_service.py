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

from google.cloud.monitoring_v3.types import snooze as gm_snooze

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "CreateSnoozeRequest",
        "ListSnoozesRequest",
        "ListSnoozesResponse",
        "GetSnoozeRequest",
        "UpdateSnoozeRequest",
    },
)


class CreateSnoozeRequest(proto.Message):
    r"""The message definition for creating a ``Snooze``. Users must provide
    the body of the ``Snooze`` to be created but must omit the
    ``Snooze`` field, ``name``.

    Attributes:
        parent (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            in which a ``Snooze`` should be created. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        snooze (google.cloud.monitoring_v3.types.Snooze):
            Required. The ``Snooze`` to create. Omit the ``name`` field,
            as it will be filled in by the API.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    snooze: gm_snooze.Snooze = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gm_snooze.Snooze,
    )


class ListSnoozesRequest(proto.Message):
    r"""The message definition for listing ``Snooze``\ s associated with the
    given ``parent``, satisfying the optional ``filter``.

    Attributes:
        parent (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            whose ``Snooze``\ s should be listed. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        filter (str):
            Optional. Optional filter to restrict results to the given
            criteria. The following fields are supported.

            - ``interval.start_time``
            - ``interval.end_time``

            For example:

            ::

                interval.start_time > "2022-03-11T00:00:00-08:00" AND
                    interval.end_time < "2022-03-12T00:00:00-08:00".
        page_size (int):
            Optional. The maximum number of results to return for a
            single query. The server may further constrain the maximum
            number of results returned in a single page. The value
            should be in the range [1, 1000]. If the value given is
            outside this range, the server will decide the number of
            results to be returned.
        page_token (str):
            Optional. The ``next_page_token`` from a previous call to
            ``ListSnoozesRequest`` to get the next page of results.
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
        number=4,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ListSnoozesResponse(proto.Message):
    r"""The results of a successful ``ListSnoozes`` call, containing the
    matching ``Snooze``\ s.

    Attributes:
        snoozes (MutableSequence[google.cloud.monitoring_v3.types.Snooze]):
            ``Snooze``\ s matching this list call.
        next_page_token (str):
            Page token for repeated calls to ``ListSnoozes``, to fetch
            additional pages of results. If this is empty or missing,
            there are no more pages.
    """

    @property
    def raw_page(self):
        return self

    snoozes: MutableSequence[gm_snooze.Snooze] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gm_snooze.Snooze,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetSnoozeRequest(proto.Message):
    r"""The message definition for retrieving a ``Snooze``. Users must
    specify the field, ``name``, which identifies the ``Snooze``.

    Attributes:
        name (str):
            Required. The ID of the ``Snooze`` to retrieve. The format
            is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/snoozes/[SNOOZE_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateSnoozeRequest(proto.Message):
    r"""The message definition for updating a ``Snooze``. The field,
    ``snooze.name`` identifies the ``Snooze`` to be updated. The
    remainder of ``snooze`` gives the content the ``Snooze`` in question
    will be assigned.

    What fields can be updated depends on the start time and end time of
    the ``Snooze``.

    - end time is in the past: These ``Snooze``\ s are considered
      read-only and cannot be updated.
    - start time is in the past and end time is in the future:
      ``display_name`` and ``interval.end_time`` can be updated.
    - start time is in the future: ``display_name``,
      ``interval.start_time`` and ``interval.end_time`` can be updated.

    Attributes:
        snooze (google.cloud.monitoring_v3.types.Snooze):
            Required. The ``Snooze`` to update. Must have the name field
            present.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The fields to update.

            For each field listed in ``update_mask``:

            - If the ``Snooze`` object supplied in the
              ``UpdateSnoozeRequest`` has a value for that field, the
              value of the field in the existing ``Snooze`` will be set
              to the value of the field in the supplied ``Snooze``.
            - If the field does not have a value in the supplied
              ``Snooze``, the field in the existing ``Snooze`` is set to
              its default value.

            Fields not listed retain their existing value.

            The following are the field names that are accepted in
            ``update_mask``:

            - ``display_name``
            - ``interval.start_time``
            - ``interval.end_time``

            That said, the start time and end time of the ``Snooze``
            determines which fields can legally be updated. Before
            attempting an update, users should consult the documentation
            for ``UpdateSnoozeRequest``, which talks about which fields
            can be updated.
    """

    snooze: gm_snooze.Snooze = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gm_snooze.Snooze,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
