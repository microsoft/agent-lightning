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

from google.cloud.trace_v2.types import trace

__protobuf__ = proto.module(
    package="google.devtools.cloudtrace.v2",
    manifest={
        "BatchWriteSpansRequest",
    },
)


class BatchWriteSpansRequest(proto.Message):
    r"""The request message for the ``BatchWriteSpans`` method.

    Attributes:
        name (str):
            Required. The name of the project where the spans belong.
            The format is ``projects/[PROJECT_ID]``.
        spans (MutableSequence[google.cloud.trace_v2.types.Span]):
            Required. A list of new spans. The span names
            must not match existing spans, otherwise the
            results are undefined.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    spans: MutableSequence[trace.Span] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=trace.Span,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
