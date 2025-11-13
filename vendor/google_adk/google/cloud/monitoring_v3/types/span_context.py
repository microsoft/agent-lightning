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
    package="google.monitoring.v3",
    manifest={
        "SpanContext",
    },
)


class SpanContext(proto.Message):
    r"""The context of a span. This is attached to an
    [Exemplar][google.api.Distribution.Exemplar] in
    [Distribution][google.api.Distribution] values during aggregation.

    It contains the name of a span with format:

    ::

        projects/[PROJECT_ID_OR_NUMBER]/traces/[TRACE_ID]/spans/[SPAN_ID]

    Attributes:
        span_name (str):
            The resource name of the span. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/traces/[TRACE_ID]/spans/[SPAN_ID]

            ``[TRACE_ID]`` is a unique identifier for a trace within a
            project; it is a 32-character hexadecimal encoding of a
            16-byte array.

            ``[SPAN_ID]`` is a unique identifier for a span within a
            trace; it is a 16-character hexadecimal encoding of an
            8-byte array.
    """

    span_name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
