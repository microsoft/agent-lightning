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

from google.cloud.aiplatform_v1.types import content as gca_content


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "GenerateSyntheticDataRequest",
        "SyntheticField",
        "SyntheticExample",
        "OutputFieldSpec",
        "TaskDescriptionStrategy",
        "GenerateSyntheticDataResponse",
    },
)


class GenerateSyntheticDataRequest(proto.Message):
    r"""Request message for DataFoundryService.GenerateSyntheticData.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        task_description (google.cloud.aiplatform_v1.types.TaskDescriptionStrategy):
            Generate data from a high-level task
            description.

            This field is a member of `oneof`_ ``strategy``.
        location (str):
            Required. The resource name of the Location to run the job.
            Format: ``projects/{project}/locations/{location}``
        count (int):
            Required. The number of synthetic examples to
            generate. For this stateless API, the count is
            limited to a small number.
        output_field_specs (MutableSequence[google.cloud.aiplatform_v1.types.OutputFieldSpec]):
            Required. The schema of the desired output,
            defined by a list of fields.
        examples (MutableSequence[google.cloud.aiplatform_v1.types.SyntheticExample]):
            Optional. A list of few-shot examples to
            guide the model's output style and format.
    """

    task_description: "TaskDescriptionStrategy" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="strategy",
        message="TaskDescriptionStrategy",
    )
    location: str = proto.Field(
        proto.STRING,
        number=1,
    )
    count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    output_field_specs: MutableSequence["OutputFieldSpec"] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message="OutputFieldSpec",
    )
    examples: MutableSequence["SyntheticExample"] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message="SyntheticExample",
    )


class SyntheticField(proto.Message):
    r"""Represents a single named field within a SyntheticExample.

    Attributes:
        field_name (str):
            Optional. The name of the field.
        content (google.cloud.aiplatform_v1.types.Content):
            Required. The content of the field.
    """

    field_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    content: gca_content.Content = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_content.Content,
    )


class SyntheticExample(proto.Message):
    r"""Represents a single synthetic example, composed of multiple
    fields. Used for providing few-shot examples in the request and
    for returning generated examples in the response.

    Attributes:
        fields (MutableSequence[google.cloud.aiplatform_v1.types.SyntheticField]):
            Required. A list of fields that constitute an
            example.
    """

    fields: MutableSequence["SyntheticField"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="SyntheticField",
    )


class OutputFieldSpec(proto.Message):
    r"""Defines a specification for a single output field.

    Attributes:
        field_name (str):
            Required. The name of the output field.
        guidance (str):
            Optional. Optional, but recommended.
            Additional guidance specific to this field to
            provide targeted instructions for the LLM to
            generate the content of a single output field.
            While the LLM can sometimes infer content from
            the field name, providing explicit guidance is
            preferred.
        field_type (google.cloud.aiplatform_v1.types.OutputFieldSpec.FieldType):
            Optional. The data type of the field.
            Defaults to CONTENT if not set.
    """

    class FieldType(proto.Enum):
        r"""The data type of the field.

        Values:
            FIELD_TYPE_UNSPECIFIED (0):
                Field type is unspecified.
            CONTENT (1):
                Arbitrary content field type.
            TEXT (2):
                Text field type.
            IMAGE (3):
                Image field type.
            AUDIO (4):
                Audio field type.
        """

        FIELD_TYPE_UNSPECIFIED = 0
        CONTENT = 1
        TEXT = 2
        IMAGE = 3
        AUDIO = 4

    field_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    guidance: str = proto.Field(
        proto.STRING,
        number=2,
    )
    field_type: FieldType = proto.Field(
        proto.ENUM,
        number=3,
        enum=FieldType,
    )


class TaskDescriptionStrategy(proto.Message):
    r"""Defines a generation strategy based on a high-level task
    description.

    Attributes:
        task_description (str):
            Required. A high-level description of the
            synthetic data to be generated.
    """

    task_description: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GenerateSyntheticDataResponse(proto.Message):
    r"""The response containing the generated data.

    Attributes:
        synthetic_examples (MutableSequence[google.cloud.aiplatform_v1.types.SyntheticExample]):
            A list of generated synthetic examples.
    """

    synthetic_examples: MutableSequence["SyntheticExample"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="SyntheticExample",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
