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
    package="google.bigtable.v2",
    manifest={
        "ResponseParams",
    },
)


class ResponseParams(proto.Message):
    r"""Response metadata proto

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        zone_id (str):
            The cloud bigtable zone associated with the
            cluster.

            This field is a member of `oneof`_ ``_zone_id``.
        cluster_id (str):
            Identifier for a cluster that represents set
            of bigtable resources.

            This field is a member of `oneof`_ ``_cluster_id``.
    """

    zone_id: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    cluster_id: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
