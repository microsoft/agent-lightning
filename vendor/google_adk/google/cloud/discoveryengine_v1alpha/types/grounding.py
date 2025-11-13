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
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GroundingFact",
        "FactChunk",
    },
)


class GroundingFact(proto.Message):
    r"""Grounding Fact.

    Attributes:
        fact_text (str):
            Text content of the fact. Can be at most 10K
            characters long.
        attributes (MutableMapping[str, str]):
            Attributes associated with the fact. Common attributes
            include ``source`` (indicating where the fact was sourced
            from), ``author`` (indicating the author of the fact), and
            so on.
    """

    fact_text: str = proto.Field(
        proto.STRING,
        number=1,
    )
    attributes: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=2,
    )


class FactChunk(proto.Message):
    r"""Fact Chunk.

    Attributes:
        chunk_text (str):
            Text content of the fact chunk. Can be at
            most 10K characters long.
        source (str):
            Source from which this fact chunk was
            retrieved. If it was retrieved from the
            GroundingFacts provided in the request then this
            field will contain the index of the specific
            fact from which this chunk was retrieved.
        index (int):
            The index of this chunk. Currently, only used
            for the streaming mode.
        source_metadata (MutableMapping[str, str]):
            More fine-grained information for the source
            reference.
    """

    chunk_text: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source: str = proto.Field(
        proto.STRING,
        number=2,
    )
    index: int = proto.Field(
        proto.INT32,
        number=4,
    )
    source_metadata: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
