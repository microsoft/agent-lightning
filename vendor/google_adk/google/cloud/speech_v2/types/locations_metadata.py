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
    package="google.cloud.speech.v2",
    manifest={
        "ModelFeature",
        "ModelFeatures",
        "ModelMetadata",
        "LanguageMetadata",
        "AccessMetadata",
        "LocationsMetadata",
    },
)


class ModelFeature(proto.Message):
    r"""Represents a singular feature of a model. If the feature is
    ``recognizer``, the release_state of the feature represents the
    release_state of the model

    Attributes:
        feature (str):
            The name of the feature (Note: the feature can be
            ``recognizer``)
        release_state (str):
            The release state of the feature
    """

    feature: str = proto.Field(
        proto.STRING,
        number=1,
    )
    release_state: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ModelFeatures(proto.Message):
    r"""Represents the collection of features belonging to a model

    Attributes:
        model_feature (MutableSequence[google.cloud.speech_v2.types.ModelFeature]):
            Repeated field that contains all features of
            the model
    """

    model_feature: MutableSequence["ModelFeature"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ModelFeature",
    )


class ModelMetadata(proto.Message):
    r"""The metadata about the models in a given region for a
    specific locale. Currently this is just the features of the
    model

    Attributes:
        model_features (MutableMapping[str, google.cloud.speech_v2.types.ModelFeatures]):
            Map of the model name -> features of that
            model
    """

    model_features: MutableMapping[str, "ModelFeatures"] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=1,
        message="ModelFeatures",
    )


class LanguageMetadata(proto.Message):
    r"""The metadata about locales available in a given region.
    Currently this is just the models that are available for each
    locale

    Attributes:
        models (MutableMapping[str, google.cloud.speech_v2.types.ModelMetadata]):
            Map of locale (language code) -> models
    """

    models: MutableMapping[str, "ModelMetadata"] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=1,
        message="ModelMetadata",
    )


class AccessMetadata(proto.Message):
    r"""The access metadata for a particular region. This can be
    applied if the org policy for the given project disallows a
    particular region.

    Attributes:
        constraint_type (google.cloud.speech_v2.types.AccessMetadata.ConstraintType):
            Describes the different types of constraints
            that are applied.
    """

    class ConstraintType(proto.Enum):
        r"""Describes the different types of constraints that can be
        applied on a region.

        Values:
            CONSTRAINT_TYPE_UNSPECIFIED (0):
                Unspecified constraint applied.
            RESOURCE_LOCATIONS_ORG_POLICY_CREATE_CONSTRAINT (1):
                The project's org policy disallows the given
                region.
        """
        CONSTRAINT_TYPE_UNSPECIFIED = 0
        RESOURCE_LOCATIONS_ORG_POLICY_CREATE_CONSTRAINT = 1

    constraint_type: ConstraintType = proto.Field(
        proto.ENUM,
        number=1,
        enum=ConstraintType,
    )


class LocationsMetadata(proto.Message):
    r"""Main metadata for the Locations API for STT V2. Currently
    this is just the metadata about locales, models, and features

    Attributes:
        languages (google.cloud.speech_v2.types.LanguageMetadata):
            Information about available locales, models,
            and features represented in the hierarchical
            structure of locales -> models -> features
        access_metadata (google.cloud.speech_v2.types.AccessMetadata):
            Information about access metadata for the
            region and given project.
    """

    languages: "LanguageMetadata" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="LanguageMetadata",
    )
    access_metadata: "AccessMetadata" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="AccessMetadata",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
