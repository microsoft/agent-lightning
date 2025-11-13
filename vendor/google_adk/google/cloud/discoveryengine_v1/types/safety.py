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
    package="google.cloud.discoveryengine.v1",
    manifest={
        "HarmCategory",
        "SafetyRating",
    },
)


class HarmCategory(proto.Enum):
    r"""Harm categories that will block the content.

    Values:
        HARM_CATEGORY_UNSPECIFIED (0):
            The harm category is unspecified.
        HARM_CATEGORY_HATE_SPEECH (1):
            The harm category is hate speech.
        HARM_CATEGORY_DANGEROUS_CONTENT (2):
            The harm category is dangerous content.
        HARM_CATEGORY_HARASSMENT (3):
            The harm category is harassment.
        HARM_CATEGORY_SEXUALLY_EXPLICIT (4):
            The harm category is sexually explicit
            content.
        HARM_CATEGORY_CIVIC_INTEGRITY (5):
            The harm category is civic integrity.
    """
    HARM_CATEGORY_UNSPECIFIED = 0
    HARM_CATEGORY_HATE_SPEECH = 1
    HARM_CATEGORY_DANGEROUS_CONTENT = 2
    HARM_CATEGORY_HARASSMENT = 3
    HARM_CATEGORY_SEXUALLY_EXPLICIT = 4
    HARM_CATEGORY_CIVIC_INTEGRITY = 5


class SafetyRating(proto.Message):
    r"""Safety rating corresponding to the generated content.

    Attributes:
        category (google.cloud.discoveryengine_v1.types.HarmCategory):
            Output only. Harm category.
        probability (google.cloud.discoveryengine_v1.types.SafetyRating.HarmProbability):
            Output only. Harm probability levels in the
            content.
        probability_score (float):
            Output only. Harm probability score.
        severity (google.cloud.discoveryengine_v1.types.SafetyRating.HarmSeverity):
            Output only. Harm severity levels in the
            content.
        severity_score (float):
            Output only. Harm severity score.
        blocked (bool):
            Output only. Indicates whether the content
            was filtered out because of this rating.
    """

    class HarmProbability(proto.Enum):
        r"""Harm probability levels in the content.

        Values:
            HARM_PROBABILITY_UNSPECIFIED (0):
                Harm probability unspecified.
            NEGLIGIBLE (1):
                Negligible level of harm.
            LOW (2):
                Low level of harm.
            MEDIUM (3):
                Medium level of harm.
            HIGH (4):
                High level of harm.
        """
        HARM_PROBABILITY_UNSPECIFIED = 0
        NEGLIGIBLE = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4

    class HarmSeverity(proto.Enum):
        r"""Harm severity levels.

        Values:
            HARM_SEVERITY_UNSPECIFIED (0):
                Harm severity unspecified.
            HARM_SEVERITY_NEGLIGIBLE (1):
                Negligible level of harm severity.
            HARM_SEVERITY_LOW (2):
                Low level of harm severity.
            HARM_SEVERITY_MEDIUM (3):
                Medium level of harm severity.
            HARM_SEVERITY_HIGH (4):
                High level of harm severity.
        """
        HARM_SEVERITY_UNSPECIFIED = 0
        HARM_SEVERITY_NEGLIGIBLE = 1
        HARM_SEVERITY_LOW = 2
        HARM_SEVERITY_MEDIUM = 3
        HARM_SEVERITY_HIGH = 4

    category: "HarmCategory" = proto.Field(
        proto.ENUM,
        number=1,
        enum="HarmCategory",
    )
    probability: HarmProbability = proto.Field(
        proto.ENUM,
        number=2,
        enum=HarmProbability,
    )
    probability_score: float = proto.Field(
        proto.FLOAT,
        number=5,
    )
    severity: HarmSeverity = proto.Field(
        proto.ENUM,
        number=6,
        enum=HarmSeverity,
    )
    severity_score: float = proto.Field(
        proto.FLOAT,
        number=7,
    )
    blocked: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
